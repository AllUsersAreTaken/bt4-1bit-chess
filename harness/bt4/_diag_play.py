"""Diagnostic: is the in-game player path picking the moves the puzzle gate says?

Step 3 found the 1.04 bpw student losing every game down to UCI_Elo 1320 while
scoring solve@1 = 0.7600 on puzzles. Two very different explanations:

  (a) the student really is ~<1000 Elo when playing greedily without search, or
  (b) `elo_match.ModelPlayer` feeds the model a different position than `Eval`
      does (history, colour, dtype) and the model is being asked the wrong
      question.

This drives the *exact* ModelPlayer call on the same 2,000 test puzzles and
compares its move with the gold move. It also plays the fp16 teacher through the
same path, which is the baseline Step 3 needs anyway: if the teacher is also
weak in games, the gap is greedy-policy play, not quantisation.

usage:
  python harness/bt4/_diag_play.py --teacher --n 200
  python harness/bt4/_diag_play.py --student results/bt4/step2_weights_full_qubo_act_ckpt.pt
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import chess
import torch

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import (_leela_all_moves, _leela_planes,
                                           _load_bt4, Lc0Vocab)   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--student", default="")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    elo = _load("elo_match", "harness/bt4/elo_match.py")
    rows = [json.loads(l) for l in PUZZLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == "test"][:args.n]

    model = _load_bt4(CKPT, "cpu").half().cuda()
    label = "fp16_teacher"
    if args.student:
        qat = _load("qat", "harness/bt4/qat.py")
        model, _ = qat.load_factors(model, args.student)
        label = pathlib.Path(args.student).stem
    model = model.half().cuda()
    vocab = Lc0Vocab(_leela_all_moves())
    player = elo.ModelPlayer(model, vocab)

    hit = 0
    for i, r in enumerate(rows):
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        gold = chess.Move.from_uci(r["moves"][1])
        mv = player(b)
        hit += int(vocab.index(b, mv) == vocab.index(b, gold))
        if i == 0:
            start = chess.Board()
            print("startpos top move:", player(start).uci(), flush=True)
    print(json.dumps(dict(label=label, n=len(rows), agree=round(hit / len(rows), 4),
                          s_per_move=round(player.seconds / max(player.moves, 1), 4))))

    # and the same model on the opening position, so a degenerate "always the
    # same move" player would be obvious
    b = chess.Board()
    planes = _leela_planes(b).unsqueeze(0).cuda().half()
    with torch.no_grad():
        pol, val, _ = model(planes)
    top = torch.topk(pol[0].float(), 5).indices.tolist()
    print("raw top-5 startpos:", [vocab.moves[i] for i in top],
          "value", val[0].float().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
