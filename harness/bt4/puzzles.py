"""Step 0b: Lichess puzzle ingest for the BT4 gate.

Reads data/bt4/lichess_db_puzzle.csv.zst once and writes a fixed, reproducible
split to data/bt4/puzzles.jsonl:

  512 calibration positions + 2,000 test positions, disjoint, deduped by FEN,
  sampled deterministically by puzzle-id hash, and validated.

Format (database.lichess.org, checked 2026-11-21): "FEN is the position before
the opponent makes their move. The position to present to the player is after
applying the first move to that FEN. The second move is the beginning of the
solution." So Moves[0] is the opponent's move and Moves[1] is the solver's
first move -- the move a 1-node policy is scored against. Every puzzle's move
list is therefore even in length (mateIn1 = [blunder, mate]).

Each row stores `moves` (the full list) and `first` (= moves[0]); the scoring
harness pushes `first` and grades on `moves[1]`, which also gives the input
plans one ply of history.

Lichess puzzles are tactical by construction; there is no quiet/strategic
stratum here and the writeup says so.
"""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import sys
from collections import Counter

import chess
import zstandard as zstd

SRC = pathlib.Path("data/bt4/lichess_db_puzzle.csv.zst")
OUT = pathlib.Path("data/bt4/puzzles.jsonl")
TRAIN_OUT = pathlib.Path("data/bt4/train.jsonl")

MIN_RATING = 1000      # below this the "key move" is often just a free capture
MAX_RATING = 3000
COLS = ["PuzzleId", "FEN", "Moves", "Rating", "Popularity", "NbPlays", "Themes"]


def hkey(pid: str) -> str:
    return hashlib.sha1(pid.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--train", type=int, default=0,
                    help="also write this many rows, disjoint from calib+test, to data/bt4/train.jsonl")
    args = ap.parse_args()

    need = args.calib + args.test
    dctx = zstd.ZstdDecompressor()
    seen_fen: set[str] = set()
    rows: list[dict] = []
    n_rows = n_rej = 0

    with open(SRC, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        for rec in csv.DictReader(text):
            n_rows += 1
            if rec["Rating"] is None:
                continue
            rating = float(rec["Rating"])
            if not (MIN_RATING <= rating <= MAX_RATING):
                n_rej += 1
                continue
            fen = rec["FEN"]
            if fen in seen_fen:
                n_rej += 1
                continue
            moves = rec["Moves"].split()
            if len(moves) < 2 or len(moves) % 2:
                n_rej += 1
                continue
            try:
                board = chess.Board(fen)
                first = chess.Move.from_uci(moves[0])
                if first not in board.legal_moves:
                    n_rej += 1
                    continue
                board.push(first)                      # the position shown to the solver
                if board.is_game_over():                # blunder already ended the game
                    n_rej += 1
                    continue
                if chess.Move.from_uci(moves[1]) not in board.legal_moves:
                    n_rej += 1
                    continue
            except ValueError:
                n_rej += 1
                continue
            seen_fen.add(fen)
            rows.append(dict(id=rec["PuzzleId"], fen=fen, first=moves[0], moves=moves,
                             rating=int(rating), n_sol=len(moves) // 2,
                             popular=int(float(rec["Popularity"] or 0)),
                             plays=int(float(rec["NbPlays"] or 0)),
                             themes=rec["Themes"] or ""))

    rows.sort(key=lambda r: hkey(r["id"]))
    picked = rows[:need]
    for i, r in enumerate(picked):
        r["split"] = "calib" if i < args.calib else "test"

    if args.train:
        # rows are hash-sorted, so everything past the eval pick is disjoint by
        # construction; a FEN dropped from the eval pick cannot reappear here.
        extra = rows[need:need + args.train]
        with open(TRAIN_OUT, "w", encoding="utf-8") as fh:
            for r in extra:
                r["split"] = "train"
                fh.write(json.dumps(r) + "\n")
        print(f"train pool {len(extra):,} rows -> {TRAIN_OUT}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r) + "\n")

    cal = [r for r in picked if r["split"] == "calib"]
    tst = [r for r in picked if r["split"] == "test"]
    print(f"csv rows {n_rows:,}  rejected {n_rej:,}  usable {len(rows):,}  picked {len(picked):,}")
    for nm, grp in (("calib", cal), ("test", tst)):
        if not grp:
            print(nm, "EMPTY")
            continue
        rat = sorted(r["rating"] for r in grp)
        def solver_is_black(r):
            b = chess.Board(r["fen"])
            b.push(chess.Move.from_uci(r["first"]))
            return b.turn == chess.BLACK
        print(f"{nm:5s} n={len(grp)} rating p10/50/90 = {rat[len(rat)//10]}/{rat[len(rat)//2]}/{rat[9*len(rat)//10]}"
              f"  solver-is-black {sum(1 for r in grp if solver_is_black(r))}")
    th = Counter(t for r in picked for t in r["themes"].split())
    print("top themes:", ", ".join(f"{k}:{v}" for k, v in th.most_common(8)))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
