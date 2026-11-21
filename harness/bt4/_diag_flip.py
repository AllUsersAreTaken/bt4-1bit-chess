"""Diagnostic: did `bit1`'s sign latents ever move, and could they?

Step 5 showed full rank lost to the factored path, but that leaves a second
explanation open: full rank is fine, and the run never actually searched the
sign patterns at all. Two things have to be true for a frozen sign pattern:

  1. the latents start far from zero, so a sign can only flip if the latent
     crosses 0 -- and Adam moves a latent by ~lr per step, not by 1.0;
  2. the latents are stored *as* the signs (|B| = 1 exactly), which is the
     worst case of (1).

`BitFull` stores `part.S`, the +-1 pattern itself, so every bit1 sign sits at
exactly |B| = 1.0. The factored path stores the arm's pre-sign latents, whose
magnitudes are set by ITQ, not by the sign.

This measures, with no training and no guesses except the LR budget:

  flips            exact. bit1's init signs are determined: `quantise_bit1` keeps
                   `s = sign(W)` and only fits h, g by ALS, so
                   sgn(checkpoint B) != sign(W) counts every sign the run moved.
  travel budget    sum of the cosine-scheduled lr over the 4,000 steps -- the
                   distance one latent can travel under Adam (step size ~ lr).
                   Reported as an estimate, because Adam's per-step step is
                   lr * m/sqrt(v), not exactly lr.
  flippable frac   share of latents inside that budget: signs a 4,000-step run
                   *can* reach. For bit1's |B| = 1 it is 0 by construction; the
                   pre-sign ratio W/(h g) and the factored |U|,|V| are measured.

usage:
  python harness/bt4/_diag_flip.py
  python harness/bt4/_diag_flip.py --ckpt results/bt4/step2_weights_bit1_4k_best.pt
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _load_bt4        # noqa: E402
from littlebit.layer import _Path                          # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
OUT = pathlib.Path("results/bt4")

# matrices sampled for the factored latent magnitudes (one _Path each)
FACTORED_SAMPLE = ("blk00.in_proj", "blk00.ff1", "blk00.ff2",
                   "blk07.in_proj", "blk07.ff2", "blk14.ff1", "blk14.ff2")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


q1 = _load("q1bit", "harness/bt4/q1bit.py")


def als(a, iters=4):
    """h, g = argmin ||a - h g^T||, the same 4-step ALS `quantise_bit1` runs."""
    g = torch.ones(a.size(1), dtype=a.dtype, device=a.device)
    h = torch.zeros(a.size(0), dtype=a.dtype, device=a.device)
    for _ in range(iters):
        h = (a @ g) / g.pow(2).sum().clamp_min(1e-12)
        g = (a.mT @ h) / h.pow(2).sum().clamp_min(1e-12)
    return h, g


def travel_budget(lr, steps, lr_min):
    """Sum of the cosine-scheduled lr: how far one latent can travel in `steps`."""
    fm = max(lr_min / lr, 1e-3)
    return lr * sum(fm + (1 - fm) * 0.5 * (1 + math.cos(math.pi * s / steps))
                    for s in range(1, steps + 1))


def pct(x, q):
    return round(float(torch.quantile(x.float().flatten(), q)), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/bt4/step2_weights_bit1_4k.pt")
    ap.add_argument("--factored-a", default="results/bt4/step2_weights_full_qubo_act_ckpt.pt",
                    help="first qubo_act checkpoint (2,000 steps, puzzles)")
    ap.add_argument("--factored-b", default="results/bt4/step2_weights_mix4000.pt",
                    help="second qubo_act checkpoint (4,000 steps, mixed pool)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--no-factored", action="store_true")
    args = ap.parse_args()

    budget = travel_budget(args.lr, args.steps, args.lr_min)
    # .half() then .float(): `quantise_bit1` signs the fp16-rounded weight, so
    # sign() must be taken on the same values or a denormal could disagree.
    model = _load_bt4(CKPT, "cpu").half().float()
    tg = q1.targets_of(model)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)["weights"]
    kind = next(iter(blob.values())).get("kind")
    if kind != "full":
        raise SystemExit(f"{args.ckpt} is a {kind!r} checkpoint, not a bit1 one")

    per, tot_b, tot_f = [], 0, 0
    for t in tg:
        w = t.get_w().detach().float().clone()
        s0 = torch.where(w >= 0, 1.0, -1.0)
        h0, g0 = als(w.abs())
        v = blob[t.label]
        s1, h1, g1 = v["B"].float(), v["h"].float(), v["g"].float()
        n = s0.numel()
        flips = int((s1 != s0).sum())
        w0 = h0[:, None] * s0 * g0[None, :]
        w1 = h1[:, None] * s1 * g1[None, :]
        den = (h1[:, None] * g1[None, :]).clamp_min(1e-12)
        latent = w / den                                   # the pre-sign latent
        per.append(dict(
            label=t.label, n_bits=n, flips=flips, flip_frac=round(flips / n, 6),
            sign_init_abs_med=round(float(s0.abs().median()), 4),
            ratio_abs_med=pct(latent.abs(), 0.5),
            ratio_flippable=round(float((latent.abs() < budget).float().mean()), 6),
            h_rel_change=round(float((h1 - h0).norm() / h0.norm().clamp_min(1e-12)), 4),
            g_rel_change=round(float((g1 - g0).norm() / g0.norm().clamp_min(1e-12)), 4),
            eff_rel_change=round(float((w1 - w0).norm() / w0.norm().clamp_min(1e-12)), 4),
        ))
        tot_b += n
        tot_f += flips
    zero = [r["label"] for r in per if r["flips"] == 0]

    rec = dict(
        ckpt=args.ckpt, steps=args.steps, lr_sign=args.lr, lr_min=args.lr_min,
        adam_travel_est=round(budget, 4),
        budget_note="sum of the cosine-scheduled lr over the run; Adam moves a "
                    "latent by ~lr per step, so this is the distance a single "
                    "latent can travel. An estimate, not a measured trajectory.",
        bit1=dict(n_matrices=len(per), n_bits=tot_b, flips=tot_f,
                  flip_frac=round(tot_f / tot_b, 8),
                  matrices_with_zero_flips=len(zero),
                  sign_init_abs_med=1.0,
                  sign_init_flippable=0.0,
                  mean_ratio_flippable=round(sum(r["ratio_flippable"] for r in per) / len(per), 6),
                  mean_eff_rel_change=round(sum(r["eff_rel_change"] for r in per) / len(per), 4),
                  mean_h_rel_change=round(sum(r["h_rel_change"] for r in per) / len(per), 4),
                  mean_g_rel_change=round(sum(r["g_rel_change"] for r in per) / len(per), 4)),
        per_matrix=per)

    if not args.no_factored:
        sample = []
        for label in FACTORED_SAMPLE:
            t = next(x for x in tg if x.label == label)
            w = t.get_w().detach().float().cuda()
            p = _Path(w, t.r, 100.0, "itq")
            u, vv = p.U.detach().abs().cpu(), p.V.detach().abs().cpu()
            sample.append(dict(label=label, r=t.r,
                               u_abs_med=pct(u, 0.5), u_abs_p10=pct(u, 0.1),
                               v_abs_med=pct(vv, 0.5), v_abs_p10=pct(vv, 0.1),
                               u_flippable=round(float((u < budget).float().mean()), 4),
                               v_flippable=round(float((vv < budget).float().mean()), 4)))
            del p, w
            torch.cuda.empty_cache()
        rec["factored_latent_scale"] = sample

        # Two runs from the *same* arm init (qubo_act, seed 0, rounds 2) with
        # different pools/lengths. Signs on which they disagree moved in at least
        # one of the two runs -- an upper bound on a single run's flips, but
        # enough to show the factored arms do move signs at all.
        a = torch.load(args.factored_a, map_location="cpu", weights_only=True)["weights"]
        b = torch.load(args.factored_b, map_location="cpu", weights_only=True)["weights"]
        # (older checkpoints predate the `kind` key, so test the factors instead)
        same = diff = 0
        for label in a:
            if "U" not in a[label] or "U" not in b.get(label, {}):
                continue
            for nm in ("U", "V"):
                same += a[label][nm].numel()
                diff += int((a[label][nm] != b[label][nm]).sum())
        rec["factored_two_runs"] = dict(
            a=pathlib.Path(args.factored_a).stem, b=pathlib.Path(args.factored_b).stem,
            n_bits=same, differing=diff,
            frac=round(diff / same, 6) if same else None,
            note="both start from the identical arm init; a differing sign moved in "
                 "at least one of the two runs (upper bound on either run's flips)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step5_flip_diag.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_matrix"}, indent=1))
    print("\n worst 8 matrices by |latent| travel:")
    for r in sorted(per, key=lambda r: r["ratio_flippable"])[:8]:
        print(f"  {r['label']:14s} flips {r['flips']:8d}/{r['n_bits']:<8d} "
              f"flip_frac {r['flip_frac']:.6f}  ratio_med {r['ratio_abs_med']:.3f}  "
              f"eff_rel_change {r['eff_rel_change']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
