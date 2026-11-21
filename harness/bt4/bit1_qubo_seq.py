"""Step 5c: sequential, output-matching QUBO for the full-rank 1-bit arm.

The first `bit1_qubo` gate fitted every matrix against Sigma of the **clean**
fp16 stream and replaced all 105 at once. That is the objective that sank
`qubo_seq` in Step 1: layer l learns to imitate the teacher on an input the
quantised model no longer produces, so the drift compounds. Its measured result
was activation error 0.5303 -> 0.4041 with no end-to-end KL recovery at all.

This runs the two corrections that were named, with no training:

  sequential   layer by layer in execution order. After each matrix is fixed the
               model is quantised in place (dense write, the exact forward
               equivalent of the binary factors), so the next layer's inputs are
               the ones the quantised model actually emits: X~.

  output-match minimise || Y_clean - W_hat X~ ||_F^2, where Y_clean = W X_clean
               is the *clean* module output. Expanding the square keeps the same
               quadratic matrix Q = diag(g) Sigma~ diag(g) and only swaps the
               linear target from W Sigma~ to the cross-covariance
               C = Y_clean X~^T = W (X_clean^T X~).  So:

                   const - 2 h_i b_i . (g . c_i) + h_i^2 b_i^T Q b_i

               i.e. the same `qubo.solve_batch` call with b_lin = g . c_i.

  ridge (optional)  Sigma~ + lambda diag(Sigma~) in Q, to see whether damping the
               ill-conditioned directions buys anything. Reported as a
               counterfactual solved from the identical moments, so it costs
               nothing extra; the installed stream uses --ridge.

One structural note: the fused MHA kernel never calls `out_proj` (verified: zero
pre-hook calls in a forward), so its real input is unhookable. Both models are
patched with an explicit SDPA forward that calls `out_proj` as a module, and the
patch is checked against the fused logits before anything is measured.

usage:
  python harness/bt4/bit1_qubo_seq.py
  python harness/bt4/bit1_qubo_seq.py --ridge 0.02 --calib 256
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
import torch.nn.functional as F

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
bq = _load("bit1_qubo", "harness/bt4/bit1_qubo.py")   # als, where_sign

als, where_sign = bq.als, bq.where_sign


def _dense_sdpa(self, x):
    """SmolgenAttention.forward with the projections as real module calls.

    Numerically the same attention as the fused kernel, but `out_proj` is invoked
    as a module so its input (the attention context) can be hooked -- which the
    fused path makes impossible. `in_proj_weight` is read at call time, so an
    in-place quantised write is picked up.
    """
    B, T, d = x.shape
    H, hd = self.num_heads, d // self.num_heads
    # (B, H, T, T): the fused kernel wants it flattened to (B*H, T, T), SDPA
    # wants it 4-D -- same tensor, and bitpath._sdpa_forward passes it this way
    bias = self._sq_bias(x)
    qkv = F.linear(x, self.mha.in_proj_weight, self.mha.in_proj_bias)
    q, k, v = qkv.split(d, dim=-1)

    def heads(t):
        return t.reshape(B, T, H, hd).transpose(1, 2)

    y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=bias)
    y = y.transpose(1, 2).reshape(B, T, d)
    return self.mha.out_proj(y), None


def patch_dense(model):
    for blk in model.transformer.layers:
        attn = blk.self_attn
        assert attn.mha.batch_first, "dense SDPA patch assumes batch_first=True"
        attn.forward = types.MethodType(_dense_sdpa, attn)


def hook_module(model, label, t):
    """The module whose input is this matrix's input, for the patched forward."""
    if label.endswith("out_proj"):
        i = int(label[3:5])
        return model.transformer.layers[i].self_attn.mha.out_proj
    if label.endswith("in_proj"):
        # the fused kernel's own in_proj module is bypassed by the patch; the
        # attention module receives exactly the same tensor
        i = int(label[3:5])
        return model.transformer.layers[i].self_attn
    return t.hook


class Pair:
    """Accumulates clean/corrupted input moments for one matrix."""

    def __init__(self, clean_mod, dirty_mod, d_in):
        self.d_in, self.n = d_in, 0
        self.S = self.Sc = self.X = None            # dirty, clean, cross
        self.xc = self.xs = None
        self.hc = clean_mod.register_forward_pre_hook(self._c)
        self.hs = dirty_mod.register_forward_pre_hook(self._s)

    def _c(self, mod, args):
        self.xc = args[0].detach().float().reshape(-1, self.d_in)

    def _s(self, mod, args):
        self.xs = args[0].detach().float().reshape(-1, self.d_in)

    def acc(self):
        if self.xc is None or self.xs is None:
            return
        if self.S is None:
            z = lambda: torch.zeros(self.d_in, self.d_in, device=self.xs.device,
                                    dtype=torch.float32)
            self.S, self.Sc, self.X = z(), z(), z()
        self.S += self.xs.mT @ self.xs
        self.Sc += self.xc.mT @ self.xc
        self.X += self.xc.mT @ self.xs
        self.n += self.xs.size(0)
        self.xc = self.xs = None

    def close(self):
        self.hc.remove()
        self.hs.remove()
        n = max(self.n, 1)
        return self.Sc / n, self.S / n, self.X / n


def out_err(W, Wq, Sc, X, S):
    """||W X_clean - Wq X~||_F / ||W X_clean||_F, from the moments only."""
    C = W @ X
    num = (W @ Sc * W).sum() - 2.0 * (C * Wq).sum() + (Wq @ S * Wq).sum()
    den = (W @ Sc * W).sum()
    return float(num.clamp_min(0).sqrt() / den.clamp_min(1e-12).sqrt())


def stream_err(W, Wq, M):
    """||(W - Wq) Z||_F / ||W Z||_F for the second moment M of Z."""
    d = W - Wq
    return float(((d @ M) * d).sum().clamp_min(0).sqrt()
                 / ((W @ M) * W).sum().clamp_min(1e-12).sqrt())


def solve(W, S, C, ridge, rounds, sweeps, restarts):
    """Row QUBOs; returns signs, h, g (scales refit after every solve)."""
    s0 = torch.where(W >= 0, 1.0, -1.0)
    h, g = als(W.abs())
    J = S + ridge * torch.diag(S).diag()
    cur = s0
    for _ in range(max(1, rounds)):
        Jq = J * g[:, None] * g[None, :]
        b_lin = C * g[None, :]
        cur, _st = lbq.solve_batch(cur, Jq, b_lin, h, None,
                                   sweeps=sweeps or None, solver="descent",
                                   gen=None, restarts=restarts)
        h, g = als(W.abs())
    return s0, cur, h, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--ridge-check", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--latent-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval", default="test")
    ap.add_argument("--tag", default="bit1_qubo_seq")
    ap.add_argument("--no-game-eval", action="store_true")
    ap.add_argument("--save-init", default="results/bt4/step5_bit1_qubo_seq_init.pt")
    args = ap.parse_args()

    rows = qat.rows_of(PUZZLES)
    cal = [r for r in rows if r["split"] == "calib"][:args.calib]
    planes = torch.stack([_leela_planes(qat.live_board(r)) for r in cal]).cuda().half()
    del rows, cal
    print(f"  {planes.size(0)} calib positions", flush=True)

    teacher = _load_bt4(CKPT, "cpu").half().cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = _load_bt4(CKPT, "cpu").half().cuda().eval()

    # ---- patch equivalence: the dense forward must reproduce the fused logits
    probe = planes[:8]
    with torch.no_grad():
        ref = teacher(probe)[0].float()
    patch_dense(teacher)
    patch_dense(student)
    with torch.no_grad():
        got = teacher(probe)[0].float()
    rel = float((ref - got).abs().max() / ref.abs().max().clamp_min(1e-12))
    print(f"  patch equivalence: max|dlogit|/max|logit| = {rel:.2e}", flush=True)
    if rel > 5e-3:
        raise SystemExit("dense SDPA patch does not reproduce the fused forward")

    # out_proj must now be callable (that is the whole point of the patch)
    fired = {"n": 0}
    probe_mod = teacher.transformer.layers[0].self_attn.mha.out_proj
    h = probe_mod.register_forward_pre_hook(lambda m, i: fired.__setitem__("n", fired["n"] + 1))
    with torch.no_grad():
        teacher(probe)
    h.remove()
    if fired["n"] == 0:
        raise SystemExit("out_proj hook still does not fire after the patch")
    print(f"  out_proj hook fires ({fired['n']} calls/forward)", flush=True)
    del probe, ref, got
    torch.cuda.empty_cache()

    tg = q1.targets_of(student)
    tgt = q1.targets_of(teacher)
    recs, parts, tot_w = [], {}, 0
    t_start = time.time()
    n_b = (planes.size(0) + args.batch - 1) // args.batch
    for k, (t, tt) in enumerate(zip(tg, tgt)):
        t0 = time.time()
        pair = Pair(hook_module(teacher, t.label, tt),
                    hook_module(student, t.label, t), t.d_in)
        for s in range(0, planes.size(0), args.batch):
            with torch.no_grad():
                teacher(planes[s:s + args.batch])
                student(planes[s:s + args.batch])
            pair.acc()
        Sc, S, X = pair.close()
        if pair.n == 0:
            raise SystemExit(f"no activations captured for {t.label}")

        W = t.get_w().detach().float()
        C = W @ X                       # Y_clean X~^T
        s0, cur, h, g = solve(W, S, C, args.ridge, args.rounds, args.sweeps,
                              args.restarts)
        wq0 = h[:, None] * s0 * g[None, :]
        # counterfactual: the same moments with a ridge, never installed
        _s0, cur_r, hr, gr = solve(W, S, C, args.ridge_check, args.rounds,
                                   args.sweeps, args.restarts)
        wq1 = h[:, None] * cur * g[None, :]
        wqr = hr[:, None] * cur_r * gr[None, :]

        rec = dict(
            label=t.label, shape=list(t.shape), n_weights=t.d_out * t.d_in,
            out_err_before=round(out_err(W, wq0, Sc, X, S), 6),
            out_err_after=round(out_err(W, wq1, Sc, X, S), 6),
            out_err_ridge=round(out_err(W, wqr, Sc, X, S), 6),
            clean_act_before=round(stream_err(W, wq0, Sc), 6),
            clean_act_after=round(stream_err(W, wq1, Sc), 6),
            dirty_act_before=round(stream_err(W, wq0, S), 6),
            dirty_act_after=round(stream_err(W, wq1, S), 6),
            frob_before=round(float((W - wq0).norm() / W.norm()), 6),
            frob_after=round(float((W - wq1).norm() / W.norm()), 6),
            frob_ridge=round(float((W - wqr).norm() / W.norm()), 6),
            qubo_flips=int((cur != s0).sum()),
            qubo_flip_frac=round(float((cur != s0).float().mean()), 6),
            secs=round(time.time() - t0, 2))
        lat = where_sign(W / (h[:, None] * g[None, :]).clamp_min(1e-12), cur).clamp(-4.0, 4.0)
        lat = where_sign(lat, cur)
        for sc in (1.0, 5.0, 10.0):
            rec[f"flippable_x{sc:g}"] = round(float(((lat.abs() / sc) < 0.22).float().mean()), 4)
        parts[t.label] = dict(kind="full", B=cur.to(torch.int8).cpu(), h=h.float().cpu(),
                              g=g.float().cpu(), latent=lat.half().cpu(),
                              d_in=t.d_in, d_out=t.d_out)
        tot_w += t.d_out * t.d_in
        recs.append(rec)
        # quantise in place: the next layer must see the corrupted stream
        t.set_w(wq1.to(torch.float16))
        del W, wq0, wq1, wqr, Sc, S, X, lat
        torch.cuda.empty_cache()
        if k % 7 == 0 or "ff2" in t.label:
            print(f"  [{args.tag}] {t.label:14s} out {rec['out_err_before']:.4f} -> "
                  f"{rec['out_err_after']:.4f} (ridge {rec['out_err_ridge']:.4f})  "
                  f"clean_act {rec['clean_act_before']:.4f} -> {rec['clean_act_after']:.4f}  "
                  f"frob {rec['frob_before']:.3f} -> {rec['frob_after']:.3f}  "
                  f"flips {rec['qubo_flip_frac']:.3f}  {rec['secs']}s", flush=True)

    bits = sum(t.d_out * t.d_in + 16 * (t.d_out + t.d_in) for t in tg)
    head = dict(arm="bit1_qubo_seq", scope="scope2", sign=True, seed=0,
                trainable=tot_w, bits=bits, bpw=round(bits / tot_w, 4),
                calib=planes.size(0), ridge=args.ridge, ridge_check=args.ridge_check,
                rounds=args.rounds, sweeps=args.sweeps or "d_in", restarts=args.restarts,
                objective="||Y_clean - W_hat X_tilde||_F (cross-covariance C = Y_clean X_tilde^T)",
                stream="sequential: each matrix quantised in place before the next capture")
    wm = lambda key: round(sum(r[key] * r["n_weights"] for r in recs) / tot_w, 6)
    agg = dict(n_matrices=len(recs), n_weights=tot_w,
               out_err_before=wm("out_err_before"), out_err_after=wm("out_err_after"),
               out_err_ridge=wm("out_err_ridge"),
               clean_act_before=wm("clean_act_before"), clean_act_after=wm("clean_act_after"),
               dirty_act_before=wm("dirty_act_before"), dirty_act_after=wm("dirty_act_after"),
               frob_before=wm("frob_before"), frob_after=wm("frob_after"),
               frob_ridge=wm("frob_ridge"),
               qubo_flip_frac=round(sum(r["qubo_flips"] for r in recs) / tot_w, 6),
               matrices_improved=int(sum(r["out_err_after"] < r["out_err_before"] for r in recs)),
               matrices_ridge_better=int(sum(r["out_err_ridge"] < r["out_err_after"] for r in recs)),
               flippable_x1=wm("flippable_x1"), flippable_x5=wm("flippable_x5"),
               flippable_x10=wm("flippable_x10"),
               capture_s=round(time.time() - t_start, 1))
    print("HEAD", json.dumps(head), flush=True)
    print("AGG", json.dumps(agg), flush=True)

    full = {lab: types.SimpleNamespace(S=v["B"].float(), h=v["h"], g=v["g"],
                                       latent=v["latent"].float())
            for lab, v in parts.items()}
    student = bp.install_full(student, full, sign=True, latent_init="ratio",
                              latent_scale=args.latent_scale)
    student = student.half().cuda().eval()
    ev = qat.Eval(args.eval)
    puzzle = ev(student)
    print("EVAL", json.dumps(puzzle), flush=True)
    game = {}
    if not args.no_game_eval:
        game = qat.GameEval(teacher)(student)
        print("GAME", json.dumps(game), flush=True)

    bases = {}
    for nm, path in (("bit1", "results/bt4/step2_qat_bit1_4k.json"),
                     ("qubo_act_factored", "results/bt4/step2_qat_full_qubo_act.json")):
        f = pathlib.Path(path)
        if f.exists():
            b = json.loads(f.read_text())["history"][0]
            bases[nm] = {k: b.get(k) for k in ("solve_top1", "solve_top5", "agree",
                                              "kl", "game_agree", "game_kl", "start")}
    prev = OUT / "step5_bit1_qubo_gate.json"
    if prev.exists():
        bases["bit1_qubo_clean_sigma"] = json.loads(prev.read_text())["zero_shot"]

    rec = dict(head=head, aggregate=agg, zero_shot={**puzzle, **game},
               baselines=bases, per_matrix=recs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"step5_{args.tag}.json").write_text(json.dumps(rec, indent=1))
    if args.save_init:
        torch.save(dict(head=head, weights=parts), args.save_init)
        print("SAVED", args.save_init, flush=True)
    print("WALL", round(time.time() - t_start, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
