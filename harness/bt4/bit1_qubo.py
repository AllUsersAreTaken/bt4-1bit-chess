"""Step 5b gate: full-rank 1-bit whose signs are chosen against the activations.

`bit1` picks `sign(W)` elementwise and then fits only row/column scales, so its
sign pattern is aligned to the raw weights and ignores what the layer sees. At
full rank that leaves ~61 % Frobenius error (the 1-bit magnitude floor), and the
pattern is frozen besides. The fix under test is the per-row QUBO initialiser:

    min_{b_i in {+-1}^d_in}  (w_i - h_i b_i . g)^T Sigma (w_i - h_i b_i . g)

Expand with `J = diag(g) Sigma diag(g)` and `b_lin_i = g . (Sigma w_i)`:

    const - 2 h_i b_i.b_lin_i + h_i^2 b_i^T J b_i

which is exactly `littlebit/qubo.solve_batch`'s form (q = 1), so each row is a
native QUBO over `d_in` bits and all `d_out` rows solve as one batch. No SVD and
no rank truncation: same 1.0 bpw as `bit1`, different objective.

What this measures, before spending any QAT time:

  act_err  ||(W - W_hat) X||_F / ||W X||_F  (activation-weighted, from Sigma)
  frob_err ||W - W_hat||_F / ||W||_F        (the 61 % floor stays; expected)
  offdiag  ||Sigma - diag(Sigma)|| / ||Sigma|| -- how much inter-channel
           correlation the QUBO can exploit at all
  zero-shot solve@1/agree/KL and held-out game_agree

The gate verdict is in `step5_bit1_qubo_gate.json`; the initialisation is saved in
`step5_bqat.load_factors` format (plus the pre-sign latents, so a later QAT run
can start with signs that are actually able to flip).

usage:
  python harness/bt4/bit1_qubo.py
  python harness/bt4/bit1_qubo.py --rounds 2 --sweeps 512
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time
import types

import chess
import numpy as np
import torch

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _leela_planes, _load_bt4   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")
OUT = pathlib.Path("results/bt4")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


q1 = _load("q1bit", "harness/bt4/q1bit.py")
bp = _load("bitpath", "harness/bt4/bitpath.py")
lbq = _load("lbq", "harness/littlebit/qubo.py")
qat = _load("qat", "harness/bt4/qat.py")


def als(a, iters=4):
    """h, g = argmin ||a - h g^T|| -- `quantise_bit1`'s exact scale fit."""
    g = torch.ones(a.size(1), dtype=a.dtype, device=a.device)
    h = torch.zeros(a.size(0), dtype=a.dtype, device=a.device)
    for _ in range(iters):
        h = (a @ g) / g.pow(2).sum().clamp_min(1e-12)
        g = (a.mT @ h) / h.pow(2).sum().clamp_min(1e-12)
    return h, g


def capture_sigmas(model, planes):
    """Sigma per Scope-2 matrix from one clean fp16 pass.

    `out_proj` is consumed by the fused MHA kernel, which does not call the
    module, so its pre-hook may never fire; its input moment is then taken from
    the attention input (same 1024 dims) -- the approximation Step 1 documents.
    """
    tg = q1.targets_of(model)
    cat, keys = {}, {}
    for t in tg:
        mod = t.hook
        if t.label.endswith("out_proj"):
            i = int(t.label[3:5])
            mod = model.transformer.layers[i].self_attn.mha
        k = (id(mod), t.d_in)
        if k not in cat:
            cat[k] = q1.SigmaCatcher(mod, t.d_in)
        keys[t.label] = k
    q1.run(model, planes)
    closed, sig, src = {}, {}, {}
    for lab, k in keys.items():
        if k not in closed:
            closed[k] = cat[k].close()
        sig[lab] = closed[k]
        src[lab] = "out_proj(same module as in_proj)" if lab.endswith("out_proj") else "own input"
    return {k: v for k, v in sig.items() if v is not None}, src


def rel_act(W, Wq, sig):
    d = W - Wq
    num = ((d @ sig) * d).sum().clamp_min(0).sqrt()
    den = ((W @ sig) * W).sum().clamp_min(0).sqrt()
    return float(num / den.clamp_min(1e-12))


def where_sign(latent, s):
    """Keep the latent's magnitude, force sgn(latent) == s exactly."""
    return torch.where(s > 0, latent.clamp_min(1e-3), latent.clamp_max(-1e-3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1,
                    help="row-solve + scale-refit alternations")
    ap.add_argument("--sweeps", type=int, default=0,
                    help="1-opt sweeps per QUBO; 0 = d_in (until stable)")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--eval", default="test")
    ap.add_argument("--tag", default="bit1_qubo_gate")
    ap.add_argument("--no-game-eval", action="store_true")
    ap.add_argument("--save-init", default="results/bt4/step5_bit1_qubo_init.pt")
    args = ap.parse_args()

    rows = qat.rows_of(PUZZLES)
    cal = [r for r in rows if r["split"] == "calib"]
    planes = torch.stack([_leela_planes(qat.live_board(r)) for r in cal]).cuda().half()

    model = _load_bt4(CKPT, "cpu").half().cuda().eval()
    tg = q1.targets_of(model)

    t0 = time.time()
    sigmas, sigma_src = capture_sigmas(model, planes)
    print(f"  captured {len(sigmas)} Sigmas from {len(cal)} calib positions "
          f"in {time.time() - t0:.1f}s", flush=True)
    del planes
    torch.cuda.empty_cache()

    recs, parts = [], {}
    tot_w = 0
    for k, t in enumerate(tg):
        t1 = time.time()
        W = t.get_w().detach().float()
        sig = sigmas[t.label].cuda()
        s0 = torch.where(W >= 0, 1.0, -1.0)
        h, g = als(W.abs())
        wq = h[:, None] * s0 * g[None, :]
        e_before, f_before = rel_act(W, wq, sig), float((W - wq).norm() / W.norm())

        def terms(h, g):
            """J and b_lin for the row QUBO given the current scales."""
            return sig * g[:, None] * g[None, :], (W * g[None, :]) @ sig.mT

        J, b_lin = terms(h, g)
        cur = s0
        for _ in range(max(1, args.rounds)):
            cur, _st = lbq.solve_batch(cur, J, b_lin, h, None,
                                       sweeps=args.sweeps or None, solver="descent",
                                       gen=None, restarts=args.restarts)
            # Exact scale refit for the chosen signs: W_hat = h s g with s fixed,
            # so (W . s) = |W| and h, g are just the ALS fit of |W|.
            h, g = als(W.abs())
            J, b_lin = terms(h, g)
        flips = int((cur != s0).sum())
        wq = h[:, None] * cur * g[None, :]
        e_after, f_after = rel_act(W, wq, sig), float((W - wq).norm() / W.norm())

        diag = sig.diagonal()
        offdiag = float((sig - torch.diag(diag)).norm() / sig.norm().clamp_min(1e-12))
        latent = where_sign(W / (h[:, None] * g[None, :]).clamp_min(1e-12),
                            cur).clamp(-4.0, 4.0)
        latent = where_sign(latent, cur)          # clamping must not flip a sign
        n = t.d_out * t.d_in
        parts[t.label] = dict(kind="full", B=cur.to(torch.int8).cpu(), h=h.float().cpu(),
                              g=g.float().cpu(), latent=latent.half().cpu(),
                              d_in=t.d_in, d_out=t.d_out)
        tot_w += n
        recs.append(dict(label=t.label, shape=list(t.shape), n_weights=n,
                         act_err_after=round(e_after, 6),
                         act_err_before_als=round(e_before, 6),
                         frob_err_after=round(f_after, 6),
                         frob_err_before_als=round(f_before, 6),
                         sigma_offdiag_frac=round(offdiag, 4),
                         qubo_flips=flips,
                         qubo_flip_frac=round(flips / n, 6),
                         secs=round(time.time() - t1, 2)))
        if k % 7 == 0 or "ff2" in t.label:
            print(f"  [{args.tag}] {t.label:14s} {str(tuple(t.shape)):14s} "
                  f"act {e_before:.4f} -> {e_after:.4f}  frob {f_before:.3f} -> "
                  f"{f_after:.3f}  offdiag {offdiag:.3f}  flips {flips / n:.4f}  "
                  f"{recs[-1]['secs']}s", flush=True)
        del W, sig, wq, latent
        torch.cuda.empty_cache()

    bits = sum(t.d_out * t.d_in + 16 * (t.d_out + t.d_in) for t in tg)
    head = dict(arm="bit1_qubo", scope="scope2", sign=True, seed=0,
                trainable=sum(t.d_out * t.d_in for t in tg),
                bits=bits, bpw=round(bits / tot_w, 4), rounds=args.rounds,
                sweeps=args.sweeps or "d_in", restarts=args.restarts,
                sigma="clean fp16 activations, 512 calib positions",
                out_proj_sigma=sigma_src.get("blk00.out_proj"))
    agg = dict(
        n_matrices=len(recs), n_weights=tot_w,
        act_err_als=round(sum(r["act_err_before_als"] * r["n_weights"] for r in recs) / tot_w, 6),
        act_err_qubo=round(sum(r["act_err_after"] * r["n_weights"] for r in recs) / tot_w, 6),
        frob_err_als=round(sum(r["frob_err_before_als"] * r["n_weights"] for r in recs) / tot_w, 6),
        frob_err_qubo=round(sum(r["frob_err_after"] * r["n_weights"] for r in recs) / tot_w, 6),
        sigma_offdiag=round(sum(r["sigma_offdiag_frac"] * r["n_weights"] for r in recs) / tot_w, 4),
        qubo_flip_frac=round(sum(r["qubo_flips"] for r in recs) / tot_w, 6),
        matrices_improved=int(sum(r["act_err_after"] < r["act_err_before_als"] for r in recs)),
    )
    print("HEAD", json.dumps(head), flush=True)
    print("AGG", json.dumps(agg), flush=True)

    # ---- install and read out zero-shot -------------------------------------
    full_parts = {lab: types.SimpleNamespace(S=v["B"].float(), h=v["h"], g=v["g"],
                                             latent=v["latent"].float())
                  for lab, v in parts.items()}
    model = bp.install_full(model, full_parts, sign=True, latent_init="ratio")
    model = model.half().cuda().eval()
    ev = qat.Eval(args.eval)
    puzzle = ev(model)
    print("EVAL", json.dumps(puzzle), flush=True)
    game = {}
    if not args.no_game_eval:
        teacher = _load_bt4(CKPT, "cpu").half().cuda().eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        game = qat.GameEval(teacher)(model)
        print("GAME", json.dumps(game), flush=True)

    bases = {}
    for nm, path in (("bit1", "results/bt4/step2_qat_bit1_4k.json"),
                     ("qubo_act_factored", "results/bt4/step2_qat_full_qubo_act.json")):
        f = pathlib.Path(path)
        if f.exists():
            b = json.loads(f.read_text())["history"][0]
            bases[nm] = {k: b.get(k) for k in ("solve_top1", "solve_top5", "agree",
                                              "kl", "game_agree", "game_kl", "start")}

    rec = dict(head=head, aggregate=agg, zero_shot={**puzzle, **game},
               baselines_note="step-0 readouts of the arms this gate is compared against",
               baselines=bases, per_matrix=recs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"step5_{args.tag}.json").write_text(json.dumps(rec, indent=1))

    if args.save_init:
        torch.save(dict(head=head, weights=parts), args.save_init)
        print("SAVED", args.save_init, flush=True)
    print("WALL", round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
