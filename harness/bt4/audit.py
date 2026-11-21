"""Step 0a: parameter audit for LeelaChessZero/BT4 (BT4-1024x15x32h-swa-6147500).

Reads the safetensors header (no weights needed) and reports the exact
linear / non-linear parameter split, so the 1.0 bpw bit budget is grounded in
the real checkpoint instead of the "191M, mostly linears" claim on the cards.

Writes results/bt4/audit.json.
"""
import collections
import json
import pathlib
import sys

HEADER = pathlib.Path("results/bt4/_header.json")
OUT = pathlib.Path("results/bt4/audit.json")

# The set the user asked to quantize first: attention q/k/v/out + MLP fc1/fc2
# in every transformer block. Everything else stays FP16.
def category(name: str) -> str:
    if name.startswith("encoder_layers."):
        blk = name.split(".")[2:]          # e.g. ['mha','q','weight']
        tail = ".".join(blk[1:]) if blk[0] in ("mha", "ffn", "ln1", "ln2") else ".".join(blk)
        if tail.startswith("smolgen."):
            return "block.smolgen"
        if tail in ("q.weight", "k.weight", "v.weight", "out.weight"):
            return "block.attn"
        if tail in ("dense1.weight", "dense2.weight"):
            return "block.mlp"
        if tail.startswith("ln"):
            return "block.layernorm"
        if tail.endswith(".bias"):
            return "block.bias" + ("." + blk[1] if len(blk) > 1 else "")
        return "block.other:" + tail
    if name.startswith("input_block."):
        return "input_block"
    if name.startswith("policy_"):
        return name.split(".")[0]
    if name.startswith("value_") or name.startswith("moves_left"):
        return name.split(".")[0]
    return "other:" + name


def main() -> int:
    hd = json.loads(HEADER.read_text())
    if "__metadata__" in hd:
        hd.pop("__metadata__")

    rows = []
    for name, info in hd.items():
        numel = 1
        for s in info["shape"]:
            numel *= s
        rows.append(dict(name=name, shape=info["shape"], numel=numel,
                         cat=category(name), ndim=len(info["shape"])))

    total = sum(r["numel"] for r in rows)
    by_cat = collections.Counter()
    cnt_cat = collections.Counter()
    for r in rows:
        by_cat[r["cat"]] += r["numel"]
        cnt_cat[r["cat"]] += 1

    # 1.0 bpw on the quantized set: 1 bit/weight + one fp16 row scale.
    quant = [r for r in rows if r["cat"] in ("block.attn", "block.mlp")]
    q_params = sum(r["numel"] for r in quant)
    q_rows = sum(r["shape"][0] for r in quant)          # one scale per output row
    q_bits = q_params + 16 * q_rows                     # sign bits + scales
    # untouched tensors in fp16
    keep = [r for r in rows if r["cat"] not in ("block.attn", "block.mlp")]
    k_params = sum(r["numel"] for r in keep)

    # block-internal tensors that are 2D but excluded (smolgen): would-be next targets
    smol = [r for r in rows if r["cat"] == "block.smolgen" and r["ndim"] == 2]
    smol_params = sum(r["numel"] for r in smol)

    blocks = {r["cat"] for r in rows}
    audit = dict(
        source="shermansiu/lc0_BT4-1024x15x32h-swa-6147500 :: model.safetensors",
        tensor_count=len(rows),
        total_params=total,
        total_params_m=round(total / 1e6, 3),
        dtype="float32 (decoded from lc0 LINEAR16)",
        by_category={k: dict(params=v, m=round(v / 1e6, 3), n_tensors=cnt_cat[k])
                     for k, v in by_cat.most_common()},
        quantize_set=dict(
            cats=["block.attn", "block.mlp"],
            params=q_params,
            m=round(q_params / 1e6, 3),
            share_of_total=round(q_params / total, 4),
            rows=q_rows,
            bits=q_bits,
            bytes_at_1bpw=round(q_bits / 8),
            mb_at_1bpw=round(q_bits / 8 / 1e6, 2),
        ),
        keep_fp16=dict(params=k_params, m=round(k_params / 1e6, 3),
                       share_of_total=round(k_params / total, 4),
                       mb=round(k_params * 2 / 1e6, 2)),
        smolgen_2d_excluded=dict(params=smol_params, m=round(smol_params / 1e6, 3),
                                 n=len(smol),
                                 mb_if_1bpw=round((smol_params + 16 * sum(r["shape"][0] for r in smol)) / 8 / 1e6, 2)),
    )
    audit["artifact_total_mb"] = round(audit["quantize_set"]["mb_at_1bpw"] + audit["keep_fp16"]["mb"], 2)
    audit["fp16_baseline_mb"] = round(total * 2 / 1e6, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=1))

    print(f"tensors {len(rows)}   total {total/1e6:.2f}M params")
    print(f"{'category':28s} {'tensors':>7s} {'params':>12s} {'share':>7s}")
    for k, v in by_cat.most_common():
        print(f"{k:28s} {cnt_cat[k]:7d} {v:12,d} {v/total*100:6.2f}%")
    q = audit["quantize_set"]
    print(f"\nquantize (attn q/k/v/out + mlp fc1/fc2): {q['m']}M = {q['share_of_total']*100:.1f}% of params"
          f" -> {q['mb_at_1bpw']} MB at 1.0 bpw ({q['rows']:,} row scales)")
    print(f"keep fp16: {audit['keep_fp16']['m']}M -> {audit['keep_fp16']['mb']} MB"
          f"  (includes {audit['smolgen_2d_excluded']['m']}M of smolgen 2D weights)")
    print(f"artifact total: {audit['artifact_total_mb']} MB   vs fp16 checkpoint {audit['fp16_baseline_mb']} MB"
          f"  = {audit['fp16_baseline_mb']/audit['artifact_total_mb']:.2f}x smaller")
    return 0


if __name__ == "__main__":
    sys.exit(main())
