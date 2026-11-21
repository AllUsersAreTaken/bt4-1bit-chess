"""Step 3: what rating does a 1.0415 bpw BT4 actually play at?

The puzzle readout (solve@1) says the student finds the teacher's move 76 % of
the time. That is a *policy-agreement* number, not a strength number: it counts
every puzzle position equally, ignores the value head, and says nothing about
what happens over a 60-move game where errors compound. So this plays games.

Two players, both reduced to `move(board) -> chess.Move`:

  * the model, greedy on the legal-masked policy (temperature 1, what the
    puzzle gate measured), fp16 on the GPU -- the deployed object;
  * Stockfish 19 with `UCI_LimitStrength`, whose `UCI_Elo` anchor is what the
    result is reported against.

The rating is *not* read off one rung. Games at several opponent settings are
pooled and a grid fit of the logistic `1/(1+10^((opp-R)/400))` is used, so a
single unlucky rung cannot decide the number. `--self-test` runs the whole
harness with Stockfish on both sides at a known gap, which is the control that
the estimator recovers what it should before it is trusted on the student.

usage:
  python harness/bt4/elo_match.py --self-test
  python harness/bt4/elo_match.py --teacher --games 8
  python harness/bt4/elo_match.py --student results/bt4/step2_weights_full_qubo_act_ckpt.pt
"""
import argparse
import importlib.util
import json
import math
import pathlib
import random
import sys
import time

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import (_leela_all_moves, _leela_planes,
                                           _load_bt4, Lc0Vocab)   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
SF = pathlib.Path("tools/stockfish/stockfish-windows-x86-64-universal.exe")
OUT = pathlib.Path("results/bt4")
NEG = torch.finfo(torch.float32).min

# Short, sound, varied openings so games do not all replay one line. Applied
# before either player thinks, two plies at a time, chosen per game by seed.
OPENINGS = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["c2c4", "g8f6", "b1c3", "e7e5"],
    ["d2d4", "g8f6", "g1f3", "d7d5"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["g1f3", "d7d5", "g2g3", "g8f6"],
    ["e2e4", "e7e5", "f1c4", "g8f6"],
]


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------- players
class ModelPlayer:
    """Greedy legal-masked policy, fp16 on the card (the shipped object)."""

    def __init__(self, model, vocab):
        self.model = model.eval()
        self.vocab = vocab
        self.dt = next(model.parameters()).dtype
        self.moves = 0
        self.seconds = 0.0

    @torch.no_grad()
    def __call__(self, board):
        # vocab index -> the real chess.Move for *this* position (castling and
        # underpromotion are encoded non-literally, so the inverse must come
        # from the board, never from decoding `vocab.moves[i]`).
        idx2mv = {}
        for mv in board.legal_moves:
            try:
                idx2mv[self.vocab.index(board, mv)] = mv
            except KeyError:
                pass
        t0 = time.time()
        planes = _leela_planes(board).unsqueeze(0).cuda().to(self.dt)
        pol, _, _ = self.model(planes)
        pol = pol[0].float()
        keys = torch.tensor(list(idx2mv), device="cuda", dtype=torch.long)
        best = int(keys[pol[keys].argmax()]) if keys.numel() else -1
        self.moves += 1
        self.seconds += time.time() - t0
        return idx2mv[best]


class _Node:
    __slots__ = ("board", "prior", "n", "w", "children")

    def __init__(self, board, prior):
        self.board, self.prior, self.n, self.w, self.children = board, prior, 0, 0.0, None

    def q(self):
        return self.w / self.n if self.n else 0.0


class PuctPlayer:
    """64-node PUCT over BT4's own policy + value (no tree reuse between moves).

    Step 3's greedy player throws the value head away and cannot see past one
    ply. This is the other half of the question: how much of the play gap is
    the *policy* and how much is the absence of search. Same network, same
    planes; the only addition is 64 simulations.
    """

    def __init__(self, model, vocab, sims=64, c_puct=1.5):
        self.model = model.eval()
        self.vocab = vocab
        self.dt = next(model.parameters()).dtype
        self.sims = sims
        self.c = c_puct
        self.moves = 0
        self.seconds = 0.0

    def _legal(self, board):
        out = {}
        for mv in board.legal_moves:
            try:
                out[mv] = self.vocab.index(board, mv)
            except KeyError:
                pass
        return out

    @torch.no_grad()
    def _eval(self, board):
        """(priors {move: p}, value in [0,1] for the side to move)."""
        legal = self._legal(board)
        planes = _leela_planes(board).unsqueeze(0).cuda().to(self.dt)
        pol, val, _ = self.model(planes)
        p = torch.softmax(pol[0].float(), -1)
        keys = torch.tensor(list(legal.values()), device="cuda", dtype=torch.long)
        p = p[keys]
        p = p / p.sum().clamp_min(1e-12)
        wdl = torch.softmax(val[0].float(), -1)
        v = float((wdl[0] - wdl[2]).clamp(-1, 1))       # win minus loss
        return {mv: float(pp) for mv, pp in zip(legal, p.tolist())}, (v + 1.0) / 2.0

    def _value_leaf(self, node):
        b = node.board
        if b.is_checkmate():
            return 0.0                                   # side to move is mated
        if b.is_game_over(claim_draw=True):
            return 0.5
        priors, v = self._eval(b)
        node.children = [_Node(b.copy(stack=True), pr) for pr in priors.values()]
        # children were built in `legal` order, which is the priors' order
        for child, mv in zip(node.children, priors):
            child.board.push(mv)
        if not node.children:
            return 0.5
        return v

    def __call__(self, board):
        t0 = time.time()
        root = _Node(board.copy(stack=True), 1.0)
        for _ in range(self.sims):
            node, path = root, [root]
            while node.children is not None:
                if not node.children:
                    break
                best, best_s = None, -1e9
                sq = math.sqrt(max(node.n, 1))
                for ch in node.children:
                    # a child stores Q from *its* side to move, one ply deeper
                    s = (1.0 - ch.q()) + self.c * ch.prior * sq / (1 + ch.n)
                    if s > best_s:
                        best, best_s = ch, s
                node = best
                path.append(node)
            v = self._value_leaf(node)                   # v: node's side to move
            for nd in reversed(path):
                nd.n += 1
                nd.w += v
                v = 1.0 - v                              # flip at every ply
        if not root.children:
            return next(iter(board.legal_moves))
        mv = max(root.children, key=lambda c: (c.n, c.prior))
        self.moves += 1
        self.seconds += time.time() - t0
        return mv.board.peek()


def sf_exe() -> str:
    """Absolute path: python-chess' popen does not resolve a relative exe on Windows."""
    p = SF.resolve()
    if not p.exists():
        raise SystemExit(f"stockfish not found at {p}")
    return str(p)


def sf_player(engine, movetime):
    def play(board):
        return engine.play(board, chess.engine.Limit(time=movetime)).move
    return play


# --------------------------------------------------------------------- games
def play_game(white, black, opening, max_ply):
    board = chess.Board()
    for uci in opening:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:      # opening line not valid here
            break
        board.push(mv)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_ply:
        player = white if board.turn == chess.WHITE else black
        mv = player(board)
        if mv is None or mv not in board.legal_moves:
            raise RuntimeError(f"illegal move {mv} in {board.fen()}")
        board.push(mv)
    return board


def adjudicate(board, engine, thresh=200, depth=12) -> str:
    """Result for a finished game, or a Chess960-style eval adjudication.

    A game that hits the ply cap returns "*" from `board.result()`, which is
    neither win nor loss: scoring it either way silently invents Elo. When the
    cap is hit, the rung's Stockfish evaluates the final position at fixed
    depth and |cp| > thresh is scored as decisive, everything else a draw.
    """
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    cp = info["score"].white().score(mate_score=10000)
    if cp is None:
        return "1/2-1/2"
    if cp > thresh:
        return "1-0"
    if cp < -thresh:
        return "0-1"
    return "1/2-1/2"


def score_of(res: str, white_is_model: bool) -> float:
    if res in ("1/2-1/2", "*"):
        return 0.5
    if white_is_model:
        return 1.0 if res == "1-0" else 0.0
    return 1.0 if res == "0-1" else 0.0


def fit_rating(games: list[dict]) -> dict:
    """Grid fit of result = 1/(1+10^((opp-R)/400)) pooled over all rungs."""
    if not games:
        return dict(rating=None, n=0)
    grid = np.arange(200.0, 3600.0, 5.0)
    obs = np.array([g["score"] for g in games])
    opp = np.array([g["opp_elo"] for g in games], dtype=float)

    def fit(o, p):
        return grid[int((((1.0 / (1.0 + 10.0 ** ((p[:, None] - grid[None, :]) / 400.0))) - o[:, None])
                         ** 2).sum(0).argmin())]

    best = float(fit(obs, opp))
    # uncertainty by bootstrapping games, not by an SSE band: with a handful of
    # decisive games the band is meaningless while the resampled spread is not.
    rng = np.random.default_rng(0)
    boots = np.array([fit(obs[i], opp[i]) for i in
                      (rng.integers(0, len(obs), len(obs)) for _ in range(2000))])
    return dict(rating=best, lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n=len(games),
                score=round(float(obs.mean()), 4),
                decisive=int(sum(1 for g in games if g["score"] != 0.5)))


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="store_true", help="play the fp16 BT4")
    ap.add_argument("--student", default="", help="step2_weights_*.pt from qat.py --save")
    ap.add_argument("--self-test", action="store_true",
                    help="Stockfish vs Stockfish control: verifies the harness + fit")
    ap.add_argument("--ladder", default="1320,1600,1900,2200")
    ap.add_argument("--games", type=int, default=8, help="games per rung (half per colour)")
    ap.add_argument("--max-ply", type=int, default=200)
    ap.add_argument("--sf-movetime", type=float, default=0.02)
    ap.add_argument("--sims", type=int, default=1, help="1 = greedy 1-node, 64 = PUCT")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]
    rng = random.Random(args.seed)
    games: list[dict] = []
    t0 = time.time()

    if args.self_test:
        # control: the same ladder, with a Stockfish of known Elo in the model's
        # seat. If the fit does not recover that anchor, nothing below is real.
        label = "sf_control"
        weak, strong = (chess.engine.SimpleEngine.popen_uci(sf_exe()),
                        chess.engine.SimpleEngine.popen_uci(sf_exe()))
        weak.configure(dict(UCI_LimitStrength=True, UCI_Elo=1500, Threads=1, Hash=16))
        model_play = sf_player(weak, args.sf_movetime)
        model_rate = 1500
    else:
        if not (args.teacher or args.student):
            ap.error("pass --teacher or --student (or --self-test)")
        label = "fp16_teacher" if args.teacher else pathlib.Path(args.student).stem
        model = _load_bt4(CKPT, "cpu").half().cuda()
        if args.student:
            qat = _load("qat", "harness/bt4/qat.py")
            model, head = qat.load_factors(model, args.student)
            print("STUDENT", json.dumps(head), flush=True)
        model = model.half().cuda()
        vocab = Lc0Vocab(_leela_all_moves())
        mp = (ModelPlayer(model, vocab) if args.sims <= 1
              else PuctPlayer(model, vocab, sims=args.sims, c_puct=args.c_puct))
        model_play = mp

    for elo in ladder:
        sf = chess.engine.SimpleEngine.popen_uci(sf_exe())
        sf.configure(dict(UCI_LimitStrength=True, UCI_Elo=elo, Threads=1, Hash=64))
        opp = sf_player(sf, args.sf_movetime)
        rung = []
        for i in range(args.games):
            model_is_white = (i % 2 == 0)            # alternate colours
            opening = OPENINGS[rng.randrange(len(OPENINGS))]
            white, black = (model_play, opp) if model_is_white else (opp, model_play)
            board = play_game(white, black, opening, args.max_ply)
            capped = not board.is_game_over(claim_draw=True)
            res = adjudicate(board, sf)
            sc = score_of(res, model_is_white)
            reason = ("checkmate" if board.is_checkmate() else
                      "stalemate" if board.is_stalemate() else
                      "adjudicated" if capped else "draw_rule")
            rung.append(dict(opp_elo=elo, score=sc, result=res, plies=board.ply(),
                             model_is_white=model_is_white, reason=reason))
            print(f"  elo {elo} game {i + 1}/{args.games}  {res}  "
                  f"{board.ply()} ply  {reason}  score {sc}", flush=True)
        sf.quit()
        w = sum(1 for g in rung if g["score"] == 1.0)
        d = sum(1 for g in rung if g["score"] == 0.5)
        print(f"RUNG {elo}  +{w} ={d} -{args.games - w - d}  "
              f"score {sum(g['score'] for g in rung) / len(rung):.3f}", flush=True)
        games += rung

    fit = fit_rating(games)
    if args.self_test:
        fit["control_anchor"] = model_rate
        fit["recovers_anchor"] = bool(fit["lo"] <= model_rate <= fit["hi"])
    rec = dict(label=label, ladder=ladder, games_per_rung=args.games, sims=args.sims,
               sf_movetime=args.sf_movetime, seed=args.seed, max_ply=args.max_ply,
               fit=fit, games=games, wall_s=round(time.time() - t0, 1))
    if not args.self_test:
        rec["model_moves"] = mp.moves
        rec["model_s_per_move"] = round(mp.seconds / max(mp.moves, 1), 4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or label
    (OUT / f"step3_elo_{tag}.json").write_text(json.dumps(rec, indent=1))
    print("FIT", json.dumps(fit))
    print("WALL", rec["wall_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
