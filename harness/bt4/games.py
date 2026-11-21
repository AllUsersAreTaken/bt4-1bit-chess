"""Step 4a: a full-game distillation pool, mixed 80/20 with the puzzle pool.

Step 3 showed the 1.04 bpw student playing `1. a3`: the fp16 teacher, through
the *same* greedy player, plays `1. d4` and reaches ~2650 Elo. Distilling on
60,000 Lichess puzzle positions only ever shows the model sharp middlegame
tactics -- never move 1, never quiet play, never an endgame -- and 128k gradient
steps on that manifold is enough to move it off the rest of chess.

So this builds the pool the user's hypothesis actually needs: real games,
sampled across their whole length, with real history.

Source: `database.lichess.org` monthly standard PGN (`data/bt4/*.pgn.zst`).
Every position keeps its **full move stack**, because
`_leela_planes()` fills its 8 history slots from `board.move_stack` -- a board
rebuilt from a bare FEN would silently lose the history and reintroduce the
same off-manifold bug in a new place.

Output, for `qat.py --pool mixed`:
  data/bt4/_planes_mixed.npy   (N,112,8,8) fp16, 80 % game + 20 % puzzle
  data/bt4/mixed.jsonl         fen + the move actually played (gold) + source
  data/bt4/_planes_games_eval.npy / games_eval.jsonl
                               held-out *games*, for full-game agreement

usage:
  python harness/bt4/games.py --n-games 80000 --puzzle-frac 0.2
"""
import argparse
import hashlib
import io
import json
import pathlib
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _leela_planes   # noqa: E402

SRC = pathlib.Path("data/bt4/lichess_2013-01.pgn.zst")
PUZZLE_PLANES = pathlib.Path("data/bt4/_planes_train.npy")
PUZZLE_TRAIN = pathlib.Path("data/bt4/train.jsonl")
OUT = pathlib.Path("data/bt4")
MIXED = OUT / "_planes_mixed.npy"
MIXED_META = OUT / "mixed.jsonl"
EVAL_PLANES = OUT / "_planes_games_eval.npy"
EVAL_META = OUT / "games_eval.jsonl"


def elo_ok(game, floor: int) -> bool:
    try:
        return (int(game.headers.get("WhiteElo", 0)) >= floor and
                int(game.headers.get("BlackElo", 0)) >= floor)
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=80000,
                    help="game positions to collect (the 80 %)")
    ap.add_argument("--puzzle-frac", type=float, default=0.2)
    ap.add_argument("--min-elo", type=int, default=1500)
    ap.add_argument("--min-ply", type=int, default=24)
    ap.add_argument("--max-ply", type=int, default=160)
    ap.add_argument("--stride", type=int, default=2, help="sample every k plies")
    ap.add_argument("--start-ply", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=40,
                    help="every k-th accepted game goes to the held-out eval set")
    ap.add_argument("--eval-games", type=int, default=60)
    args = ap.parse_args()

    t0 = time.time()
    gcap = args.n_games
    game_planes = np.zeros((gcap, 112, 8, 8), dtype=np.float16)
    game_meta: list[dict] = []
    eval_planes_l: list[np.ndarray] = []
    eval_meta: list[dict] = []

    dctx = zstd.ZstdDecompressor()
    n_game = n_pos = n_seen = n_short = n_acc = 0
    with open(SRC, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
        while n_game < gcap:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            n_seen += 1
            if not elo_ok(game, args.min_elo):
                continue
            moves = list(game.mainline_moves())
            if not (args.min_ply <= len(moves) <= args.max_ply):
                n_short += 1
                continue
            is_eval = (n_acc % args.eval_every == args.eval_every - 1
                       and len(eval_meta) < args.eval_games * 40)
            board = game.board()                 # keeps the real move stack
            taken = 0
            for p, mv in enumerate(moves):
                if p >= args.start_ply and (p - args.start_ply) % args.stride == 0:
                    pl = _leela_planes(board).numpy()
                    if is_eval:
                        eval_planes_l.append(pl)
                        eval_meta.append(dict(fen=board.fen(), gold=mv.uci(),
                                              ply=p, split="eval"))
                    elif n_game + taken < gcap:
                        game_planes[n_game + taken] = pl
                        game_meta.append(dict(fen=board.fen(), gold=mv.uci(),
                                              ply=p, src="game", split="train"))
                        taken += 1
                board.push(mv)
            n_game += taken
            n_pos += taken
            n_acc += 1
            if n_seen % 200 == 0:
                print(f"  {n_seen:,} games read, {n_pos:,} positions, "
                      f"{time.time() - t0:.0f}s", flush=True)

    game_planes = game_planes[:n_pos]
    print(f"game positions {n_pos:,} from {n_seen:,} PGN games "
          f"({n_short:,} too short/skipped)  eval {len(eval_meta):,}", flush=True)

    # ---- 20 % puzzles, drawn from the Step 2 pool with a fixed seed
    n_puz = int(round(n_pos * args.puzzle_frac / (1 - args.puzzle_frac)))
    puz = np.load(PUZZLE_PLANES, mmap_mode="r")
    puz_rows = [json.loads(l) for l in
                PUZZLE_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = np.random.default_rng(0).choice(puz.shape[0], size=n_puz, replace=False)
    puz_sel = np.asarray(puz[np.sort(idx)])

    mixed = np.concatenate([game_planes, puz_sel], axis=0)
    del game_planes
    # the puzzle row's `fen` is the position *before* the opponent's move; the
    # scored position (and therefore the legal mask) is one ply later.
    def live_fen(r):
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        return b.fen()

    meta = game_meta + [dict(fen=live_fen(puz_rows[i]), gold=puz_rows[i]["moves"][1],
                             ply=1, src="puzzle", split="train")
                        for i in np.sort(idx)]
    order = np.random.default_rng(1).permutation(mixed.shape[0])
    mixed = mixed[order]
    meta = [meta[i] for i in order]

    np.save(MIXED, mixed)
    with open(MIXED_META, "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    if eval_planes_l:
        np.save(EVAL_PLANES, np.stack(eval_planes_l))
        with open(EVAL_META, "w", encoding="utf-8") as fh:
            for m in eval_meta:
                fh.write(json.dumps(m) + "\n")

    plies = np.array([m["ply"] for m in meta])
    srcs = np.array([m["src"] for m in meta])
    print(f"mixed {mixed.shape} -> {MIXED}  ({mixed.nbytes / 2**30:.2f} GiB)")
    print(f"  game {int((srcs == 'game').sum()):,}  puzzle {int((srcs == 'puzzle').sum()):,}"
          f"  ply p10/50/90/99 = {np.percentile(plies, 10):.0f}/"
          f"{np.percentile(plies, 50):.0f}/{np.percentile(plies, 90):.0f}/"
          f"{np.percentile(plies, 99):.0f}")
    print(f"  eval {len(eval_meta):,} positions -> {EVAL_PLANES}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
