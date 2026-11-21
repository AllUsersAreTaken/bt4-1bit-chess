"""Step 0d: the 1.0 bpw bit budget, computed on the loaded module tree.

audit.py counts what the .pb.gz *stores*; this counts what inference actually
*uses*. They differ because lc0's file carries training-only heads and 15
redundant copies of the shared GAB bank.

For each quantization scope it reports 1-bit packed size (sign bits + one fp16
row scale per 2-D weight) plus the fp16 cost of everything left alone.

Writes results/bt4/budget.json.
"""
import json
import pathlib
import sys

import torch
import torch.nn as nn

from chessformer_lens.adapters.lc0 import _load_bt4

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
OUT = pathlib.Path("results/bt4/budget.json")

BLOCK_NAMED = ("self_attn.mha.in_proj_weight", "self_attn.mha.out_proj",
               "linear1", "linear2")


def name_of(mod_name: str) -> str:
    return mod_name


def collect(model) -> list[dict]:
    """Every distinct trainable tensor once (the shared GAB bank appears 15x)."""
    seen, rows = set(), []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        rows.append(dict(name=name, numel=p.numel(), shape=tuple(p.shape),
                         ndim=p.ndim))
    return rows


def scope(row, kind: str) -> bool:
    n, nd = row["name"], row["ndim"]
    if nd != 2:                      # norms, biases, gates stay fp16
        return False
    in_block = n.startswith("transformer.layers.")
    if kind == "named":              # what Step 1 quantizes first
        return in_block and any(k in n for k in BLOCK_NAMED)
    if kind == "blocks":             # + smolgen
        return in_block
    if kind == "all":                # + input block + policy/value heads
        return True
    raise ValueError(kind)


def main() -> int:
    model = _load_bt4(CKPT, "cpu")
    rows = collect(model)
    total = sum(r["numel"] for r in rows)

    out = dict(ckpt=CKPT, used_params=total, used_m=round(total / 1e6, 3),
               n_distinct_tensors=len(rows), scopes={})
    print(f"used params {total/1e6:.2f}M in {len(rows)} distinct tensors")
    print(f"{'scope':10s} {'quant':>10s} {'1bpw MB':>8s} {'keep fp16 MB':>12s} {'artifact MB':>11s} {'vs fp16':>8s}")
    for kind in ("named", "blocks", "all"):
        q = [r for r in rows if scope(r, kind)]
        keep = [r for r in rows if not scope(r, kind)]
        qp = sum(r["numel"] for r in q)
        qrows = sum(r["shape"][0] for r in q)
        mb = (qp + 16 * qrows) / 8 / 1e6
        kp = sum(r["numel"] for r in keep)
        kmb = kp * 2 / 1e6
        art = mb + kmb
        out["scopes"][kind] = dict(
            quant_params=qp, quant_m=round(qp / 1e6, 3),
            share=round(qp / total, 4), quant_mb=round(mb, 2),
            row_scales=qrows, keep_params=kp, keep_mb=round(kmb, 2),
            artifact_mb=round(art, 2),
            shrink_vs_fp16=round(total * 2 / 1e6 / art, 2),
            per_tensor={r["name"]: r["numel"] for r in q} if kind == "named" else None)
        print(f"{kind:10s} {qp/1e6:9.2f}M {mb:8.2f} {kmb:12.1f} {art:11.1f} {total*2/1e6/art:7.2f}x")

    out["fp16_all_mb"] = round(total * 2 / 1e6, 1)
    out["note"] = ("1.0 bpw = 1 sign bit/weight + one fp16 scale per output row; "
                  "2-D weights in the named/blocks/all scopes only. "
                  "Row scales would need requantising after every QAT step.")
    OUT.write_text(json.dumps(out, indent=1))
    print("fp16 (used weights):", out["fp16_all_mb"], "MB")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
