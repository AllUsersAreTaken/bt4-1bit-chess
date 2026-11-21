"""Step 0c: FP16 BT4 baseline on Lichess puzzles (the reference every 1-bit arm
is scored against).

Reports, for a split of data/bt4/puzzles.jsonl:
  - searchless (1-node) puzzle solve rate: policy top-1 (and top-5) == the
    solution's first move, after masking to legal moves
  - legal-move policy entropy (nats), normalised entropy, and mean top-1 prob
  - WDL readout: argmax class distribution, mean q = P(win) - P(loss)
  - 4060 throughput in positions/second at the chosen batch size

It also caches the FP16 policy logits (N, 4288, fp16) and value logits (N, 3)
so every quantized arm is compared against the exact same reference numbers
instead of a re-computation.

WDL *accuracy* needs ground-truth outcomes, which the Lichess puzzle file does
not carry: this script reports the WDL readout and its agreement structure, not
an accuracy, and results/bt4/STEP0.md says so explicitly.

usage:
  python harness/bt4/base.py --split test --batch 64
  python harness/bt4/base.py --split test --limit 8 --batch 8 --no-cache   # smoke
"""
import argparse
import json
import math
import pathlib
import time

import chess
import numpy as np
import torch

from chessformer_lens.adapters.lc0 import (_leela_all_moves, _leela_planes,
                                           _load_bt4, Lc0Vocab)

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")
CACHE = pathlib.Path("results/bt4/_cache")


def load_split(split: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in PUZZLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_split(args.split, args.limit)
    if not rows:
        print("empty split", args.split)
        return 1

    t0 = time.time()
    model = _load_bt4(args.ckpt, "cpu").half().to(dev).eval()
    load_s = time.time() - t0
    vocab = Lc0Vocab(_leela_all_moves())
    n_vocab = len(vocab.moves)

    # Moves[0] is the opponent's blunder; the position shown to the solver is
    # after it, and Moves[1] is the solver's first move (see puzzles.py).
    boards, gold = [], []
    for r in rows:
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        boards.append(b)
        gold.append(vocab.index(b, chess.Move.from_uci(r["moves"][1])))
    planes = torch.stack([_leela_planes(b) for b in boards]).to(dev).half()
    masks = torch.stack([vocab.legal_mask(b) for b in boards]).to(dev)

    n = len(rows)
    top1 = np.zeros(n, dtype=np.int64)
    gold_rank = np.zeros(n, dtype=np.int64)
    entropy = np.zeros(n, dtype=np.float64)
    top1_prob = np.zeros(n, dtype=np.float64)
    wdl = np.zeros((n, 3), dtype=np.float64)
    logits_fp16 = np.zeros((n, n_vocab), dtype=np.float16) if not args.no_cache else None
    legal_counts = masks.sum(1).cpu().numpy()

    played = 0
    t_infer = 0.0
    with torch.no_grad():
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            x = planes[s:e]
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            pol, val, _ = model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t_infer += time.time() - t
            m = masks[s:e]
            masked = pol.float().masked_fill(~m, float("-inf"))
            lp = torch.log_softmax(masked, dim=-1)
            p = lp.exp()
            top1[s:e] = masked.argmax(-1).cpu().numpy()
            top1_prob[s:e] = p.max(-1).values.cpu().numpy()
            # rank of the solution move among legal moves
            gold_t = torch.tensor(gold[s:e], device=dev)
            greater = (masked > masked.gather(1, gold_t[:, None])).sum(1)
            gold_rank[s:e] = greater.cpu().numpy()
            ent = -torch.where(m, p * lp, torch.zeros_like(lp)).sum(-1)
            entropy[s:e] = ent.cpu().numpy()
            wdl[s:e] = torch.softmax(val.float(), dim=-1).cpu().numpy()
            if logits_fp16 is not None:
                logits_fp16[s:e] = pol.float().cpu().numpy().astype(np.float16)
            played += e - s

    solved = top1 == np.array(gold)
    top5 = gold_rank < 5
    q = wdl[:, 0] - wdl[:, 2]
    res = dict(
        split=args.split, n=int(n), ckpt=args.ckpt, device=dev,
        dtype="fp16", batch=args.batch,
        load_s=round(load_s, 2), infer_s=round(t_infer, 2),
        pos_per_s=round(played / t_infer, 2) if t_infer else None,
        ms_per_pos=round(1000 * t_infer / played, 2) if played else None,
        solve_top1=round(float(solved.mean()), 4),
        solve_top5=round(float(top5.mean()), 4),
        legal_moves_mean=round(float(legal_counts.mean()), 2),
        legal_moves_min=int(legal_counts.min()), legal_moves_max=int(legal_counts.max()),
        entropy_nats=round(float(entropy.mean()), 4),
        entropy_norm=round(float(np.mean(entropy / np.log(np.maximum(legal_counts, 2)))), 4),
        top1_prob=round(float(top1_prob.mean()), 4),
        wdl_argmax=dict(zip(("win", "draw", "loss"),
                            np.bincount(wdl.argmax(1), minlength=3).tolist())),
        wdl_mean=dict(win=round(float(wdl[:, 0].mean()), 4),
                      draw=round(float(wdl[:, 1].mean()), 4),
                      loss=round(float(wdl[:, 2].mean()), 4)),
        mean_q=round(float(q.mean()), 4),
        mean_q_solved=round(float(q[solved].mean()), 4) if solved.any() else None,
        mean_q_unsolved=round(float(q[~solved].mean()), 4) if (~solved).any() else None,
    )

    out = pathlib.Path(args.out) if args.out else pathlib.Path(f"results/bt4/fp16_base_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    if logits_fp16 is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(CACHE / f"policy_fp16_{args.split}.npy", logits_fp16)
        np.save(CACHE / f"value_fp16_{args.split}.npy", wdl.astype(np.float32))
        np.save(CACHE / f"gold_idx_{args.split}.npy", np.array(gold))
    for k, v in res.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
