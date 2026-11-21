"""Is rank d/2 itself a wall on BT4? Exact SVD truncation, zero training.

The Step 4 control (continuous low-rank, 1,000 steps) is confounded: with Adam
a continuous factor moves ~lr per step, so in 1,000 steps at lr 1e-4 -> 1e-5 the
factors drift ~0.1, while an STE latent moving 0.01 can flip a sign and change
an effective weight by 2.0. So it cannot separate "rank d/2 has no capacity"
from "continuous factors just optimise slowly at that LR".

This settles capacity with no optimiser at all: replace every Scope-2 matrix
with its **exact rank-r SVD truncation** (r = round(d_out*d_in/(d_out+d_in)),
the same rank the 1.04 bpw path uses), in fp16, and measure what is left.

  exact SVD is the *best* rank-r approximation in Frobenius norm, so this is an
  upper bound on what any rank-r model could do at any precision.

usage:
  python harness/bt4/_diag_svd.py
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _load_bt4   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scope2(model):
    """(label, module, attribute) for the same 105 matrices the paths cover."""
    out = []
    for i, blk in enumerate(model.transformer.layers):
        p, attn = f"blk{i:02d}", blk.self_attn
        for nm in ("sm1", "sm2", "sm3"):
            out.append((f"{p}.{nm}", getattr(attn, nm), "weight"))
        out.append((f"{p}.in_proj", attn.mha, "in_proj_weight"))
        out.append((f"{p}.out_proj", attn.mha.out_proj, "weight"))
        out.append((f"{p}.ff1", blk.linear1, "weight"))
        out.append((f"{p}.ff2", blk.linear2, "weight"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.0,
                    help="rank as a multiple of the 1.04 bpw rank r")
    args = ap.parse_args()
    qat = _load("qat", "harness/bt4/qat.py")

    teacher = _load_bt4(CKPT, "cpu").half().cuda().eval()
    ev = qat.Eval("test")                 # fp16 reference logits come from disk
    gev = qat.GameEval(teacher)           # full-game refs, from the *untouched* teacher

    student = _load_bt4(CKPT, "cpu").half().cuda()

    dt = torch.float16
    rels, ranks, bits = [], 0, 0
    for label, mod, attr in scope2(student):
        W = getattr(mod, attr).data.float()
        d_out, d_in = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = max(1, round(args.mult * d_out * d_in / (d_out + d_in)))
        r = min(r, S.numel())
        Wr = (U[:, :r] * S[:r]) @ Vh[:r]
        rels.append(float((W - Wr).norm() / W.norm()))
        ranks += (d_in + d_out) * r
        bits += 16 * (d_in + d_out) * r
        with torch.no_grad():
            getattr(mod, attr).data.copy_(Wr.to(dt))

    n_w = sum(W.numel() for _, m, a in scope2(student)
              for W in [getattr(m, a).data])

    with torch.no_grad():
        puzzle = ev(student)
        game = gev(student)

    rec = dict(label=f"exact_svd_rank_{args.mult:g}x_r_fp16", mult=args.mult,
               ranks_note="r = mult * round(d_out*d_in/(d_out+d_in)); 1x is the 1.04 bpw path",
               n_matrices=len(rels), n_weights=n_w,
               mean_rel_fro_err=round(float(np.mean(rels)), 4),
               max_rel_fro_err=round(float(np.max(rels)), 4),
               factor_bits=bits, factor_bpw=round(bits / n_w, 2),
               **puzzle, **game)
    out = pathlib.Path(f"results/bt4/step4_svd_rank{args.mult:g}x.json")
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
