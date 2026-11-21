"""Step 2: distill the 1.0 bpw BT4 body from its fp16 teacher.

Student: the Step 1 arm's factors installed as factored `BitPath` modules, so
what trains is exactly the 1.0415 bpw object (signs + three scales), not a
dense proxy.

Loss, all against the frozen fp16 teacher on the same position:
  policy  KL(teacher || student) over the legal-move vocabulary (temperature 1)
  value   cross-entropy of the student WDL logits on the teacher's probabilities
  hidden  relative MSE of the last block's output (the residual stream the
          heads read, which is where Step 1 measured the 1.19 drift)

The teacher stays resident: BT4 is 370 MB in fp16 (0.68 GiB peak at batch 256),
so co-residency on the 4060 costs nothing and avoids per-batch host copies.

usage:
  python harness/bt4/qat.py --prep                     # build the plane cache
  python harness/bt4/qat.py --arm itq --steps 200      # pilot
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
import types

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import (_leela_all_moves, _leela_planes,
                                           _load_bt4, Lc0Vocab)   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
TRAIN = pathlib.Path("data/bt4/train.jsonl")
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")
PLANES = pathlib.Path("data/bt4/_planes_train.npy")
MIXED_PLANES = pathlib.Path("data/bt4/_planes_mixed.npy")
MIXED_META = pathlib.Path("data/bt4/mixed.jsonl")
GAME_EVAL = pathlib.Path("data/bt4/games_eval.jsonl")
GAME_EVAL_PLANES = pathlib.Path("data/bt4/_planes_games_eval.npy")
CACHE = pathlib.Path("results/bt4/_cache")
OUT = pathlib.Path("results/bt4")
NEG = torch.finfo(torch.float32).min


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


q1 = _load("q1bit", "harness/bt4/q1bit.py")
bp = _load("bitpath", "harness/bt4/bitpath.py")


def rows_of(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def live_board(r):
    b = chess.Board(r["fen"])
    b.push(chess.Move.from_uci(r["first"]))
    return b


def pos_board(r):
    """The position a pool row is actually scored on.

    Puzzle rows carry the pre-move FEN plus `first`; full-game rows carry the
    FEN of the position itself (their history lives in the cached planes).
    """
    b = chess.Board(r["fen"])
    if r.get("first"):
        b.push(chess.Move.from_uci(r["first"]))
    return b


def prep(args) -> int:
    rows = rows_of(TRAIN)
    if args.limit:
        rows = rows[:args.limit]
    arr = np.zeros((len(rows), 112, 8, 8), dtype=np.float16)
    t0 = time.time()
    for i, r in enumerate(rows):
        arr[i] = _leela_planes(live_board(r)).numpy()
        if i % 5000 == 0:
            print(f"  {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    np.save(PLANES, arr)
    print(f"{arr.shape} fp16 -> {PLANES}  ({arr.nbytes / 2**20:.0f} MiB, {time.time() - t0:.0f}s)")
    return 0


class Masks:
    """Legal-move masks for the training pool, filled lazily (257 MB at 60k)."""

    def __init__(self, rows, vocab):
        self.rows, self.vocab = rows, vocab
        self.arr = np.zeros((len(rows), len(vocab.moves)), dtype=np.bool_)
        self.have = np.zeros(len(rows), dtype=np.bool_)

    def get(self, idx):
        for i in idx.tolist():
            if not self.have[i]:
                self.arr[i] = self.vocab.legal_mask(pos_board(self.rows[i])).numpy()
                self.have[i] = True
        return torch.from_numpy(self.arr[idx]).cuda()


class Eval:
    """Fixed 2,000-puzzle readout: solve@1/5, agreement, KL against fp16."""

    def __init__(self, split="test"):
        rows = [r for r in rows_of(PUZZLES) if r["split"] == split]
        self.vocab = Lc0Vocab(_leela_all_moves())
        boards = [live_board(r) for r in rows]
        self.planes = torch.stack([_leela_planes(b) for b in boards]).cuda().half()
        self.masks = torch.stack([self.vocab.legal_mask(b) for b in boards]).cuda()
        self.gold = torch.tensor([self.vocab.index(b, chess.Move.from_uci(r["moves"][1]))
                                  for b, r in zip(boards, rows)], device="cuda")
        self.ref = torch.tensor(np.load(CACHE / f"policy_fp16_{split}.npy").astype(np.float32),
                                device="cuda")
        self.lp_t = F.log_softmax(self.ref.masked_fill(~self.masks, NEG), -1)

    @torch.no_grad()
    def __call__(self, model, batch=64):
        """fp32 on purpose: the readout must be the number Step 1 reported.

        A NaN/inf logit makes every `>` comparison False, which would score as
        "rank 0" and fake a perfect solve@1 — so non-finite rows are counted and
        excluded instead of silently winning.
        """
        n = self.planes.size(0)
        dt = next(model.parameters()).dtype       # fp32 while training, fp16 for a guide check
        logits = torch.empty((n, self.ref.size(1)), dtype=torch.float32, device="cuda")
        t0 = time.time()
        for s in range(0, n, batch):
            pol, _, _ = model(self.planes[s:s + batch].to(dt))
            logits[s:s + batch] = pol.float()
        infer = time.time() - t0
        ok = torch.isfinite(logits).all(1)
        logits = torch.where(ok[:, None], logits, torch.zeros_like(logits))
        m = logits.masked_fill(~self.masks, NEG)
        rank = (m > m.gather(1, self.gold[:, None])).sum(1)
        lp_q = F.log_softmax(m, -1)
        kl = torch.where(self.masks, self.lp_t.exp() * (self.lp_t - lp_q),
                         torch.zeros((), device="cuda")).sum(-1)
        okf = ok.float()
        return dict(solve_top1=round(float(((rank == 0).float() * okf).sum() / okf.sum().clamp_min(1)), 4),
                    solve_top5=round(float(((rank < 5).float() * okf).sum() / okf.sum().clamp_min(1)), 4),
                    agree=round(float(((m.argmax(1) == self.ref.argmax(1)).float() * okf).sum()
                                     / okf.sum().clamp_min(1)), 4),
                    kl=round(float(kl[ok].mean()) if bool(ok.any()) else float("nan"), 4),
                    nonfinite=int((~ok).sum()),
                    pos_per_s=round(n / infer, 1))


class GameEval:
    """Held-out real-game positions: does the student rank moves like the teacher
    *off* the puzzle manifold, and what does it actually play on move 1?

    This is the readout Step 3's failure needs: puzzle `agree` was 0.61 while
    the model played `1. a3`. Full-game agreement plus the move-1 string makes
    both visible in every EVAL line.
    """

    def __init__(self, teacher):
        rows = rows_of(GAME_EVAL)
        self.vocab = Lc0Vocab(_leela_all_moves())
        self.planes = torch.tensor(np.load(GAME_EVAL_PLANES), device="cuda").half()
        boards = [chess.Board(r["fen"]) for r in rows]
        self.masks = torch.stack([self.vocab.legal_mask(b) for b in boards]).cuda()
        self.gold = torch.tensor([self.vocab.index(b, chess.Move.from_uci(r["gold"]))
                                  for b, r in zip(boards, rows)], device="cuda")
        ref = []
        with torch.no_grad():
            for s in range(0, self.planes.size(0), 128):
                pol, _, _ = teacher(self.planes[s:s + 128])
                ref.append(pol.float())
        self.ref = torch.cat(ref).cuda()
        self.lp_t = F.log_softmax(self.ref.masked_fill(~self.masks, NEG), -1)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, model, batch=64):
        n = self.planes.size(0)
        dt = next(model.parameters()).dtype
        logits = torch.empty((n, self.ref.size(1)), dtype=torch.float32, device="cuda")
        for s in range(0, n, batch):
            pol, _, _ = model(self.planes[s:s + batch].to(dt))
            logits[s:s + batch] = pol.float()
        ok = torch.isfinite(logits).all(1)
        m = logits.masked_fill(~self.masks, NEG)
        agree = (m.argmax(1) == self.lp_t.argmax(1)).float()
        gold = (m.argmax(1) == self.gold).float()
        lp_q = F.log_softmax(m, -1)
        kl = torch.where(self.masks, self.lp_t.exp() * (self.lp_t - lp_q),
                         torch.zeros((), device="cuda")).sum(-1)
        b = chess.Board()                      # the exact Step 3 failure
        p1, _, _ = model(_leela_planes(b).unsqueeze(0).cuda().to(dt))
        p1 = p1[0].float().masked_fill(~self.vocab.legal_mask(b).cuda(), NEG)
        start = self.vocab.moves[int(p1.argmax())]
        return dict(game_agree=round(float(agree.mean()), 4),
                    game_kl=round(float(kl[ok].mean()) if bool(ok.any()) else float("nan"), 4),
                    game_gold=round(float(gold.mean()), 4), start=start)


def compute_loss(spol, sval, shid, tpol, tval, thid, mask, args):
    lp_t = F.log_softmax(tpol.masked_fill(~mask, NEG), -1)
    lp_s = F.log_softmax(spol.masked_fill(~mask, NEG), -1)
    kl = torch.where(mask, lp_t.exp() * (lp_t - lp_s), torch.zeros((), device=mask.device))
    policy = kl.sum(-1).mean()
    value = -(F.softmax(tval.float(), -1) * F.log_softmax(sval.float(), -1)).sum(-1).mean()
    num = (shid - thid).pow(2).sum(-1).mean(-1)
    den = thid.pow(2).sum(-1).mean(-1).clamp_min(1e-6)
    hidden = (num / den).mean()
    loss = args.w_policy * policy + args.w_value * value + args.w_hidden * hidden
    return loss, dict(policy=round(float(policy), 4), value=round(float(value), 4),
                      hidden=round(float(hidden), 5))


def save_factors(model, tag, head) -> pathlib.Path:
    """Persist the trained signs (1 byte each, exact) + fp32 scales.

    Signs are stored as int8 rather than fp16 so the checkpoint is bit-exact:
    they are the thing QAT actually moved, and every fp32/fp16 forward derives
    from sgn(U), sgn(V).
    """
    w = {}
    for nm, p in bp.bitpaths(model):
        if p.kind == "full":
            w[nm] = dict(kind="full",
                         B=(p.B.detach().cpu().sign().to(torch.int8) if p.signed
                            else p.B.detach().half().cpu()),
                         h=p.h.detach().float().cpu(), g=p.g.detach().float().cpu(),
                         d_in=p.d_in, d_out=p.d_out)
            continue
        # int8 signs are exact for a binary path; a continuous (control) path
        # has to keep its real factors or the checkpoint is a different model.
        w[nm] = dict(kind="factored",
                     U=(p.U.detach().cpu().sign().to(torch.int8) if p.signed
                        else p.U.detach().half().cpu()),
                     V=(p.V.detach().cpu().sign().to(torch.int8) if p.signed
                        else p.V.detach().half().cpu()),
                     h=p.h.detach().float().cpu(), g=p.g.detach().float().cpu(),
                     l=p.l.detach().float().cpu(),
                     d_in=p.d_in, d_out=p.d_out, r=p.r)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"step2_weights_{tag}.pt"
    torch.save(dict(head=head, weights=w), f)
    return f


def load_factors(model, path, latent_scale=1.0):
    """Rebuild the trained student on a fresh model: returns (model, head)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    w = blob["weights"]
    first = next(iter(w.values()))
    # a checkpoint is already in its trained parameterisation, so it is loaded
    # verbatim; int8 storage means binary, fp16 means a continuous control
    # older full-rank inits (the bit1_qubo gates) were written before `kind` was
    # set, so the format is detected from the tensors themselves
    if first.get("kind") == "full" or ("B" in first and "U" not in first):
        # a checkpoint written by the bit1_qubo gate also carries the pre-sign
        # latents, so a QAT run resumes with signs that can still move
        parts = {nm: types.SimpleNamespace(
            S=v["B"].float(), h=v["h"], g=v["g"],
            **({"latent": v["latent"].float()} if "latent" in v else {}))
            for nm, v in w.items()}
        return bp.install_full(model, parts, sign=first["B"].dtype == torch.int8,
                               latent_init="ratio" if "latent" in first else "sign",
                               latent_scale=latent_scale), blob["head"]
    paths = {nm: types.SimpleNamespace(U=v["U"].float(), V=v["V"].float(),
                                      h=v["h"], g=v["g"], l=v["l"])
             for nm, v in w.items()}
    return bp.install(model, paths, sign=first["U"].dtype == torch.int8,
                      binarise_init=False), blob["head"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default="itq",
                    choices=["itq", "qubo_act", "qubo_seq", "bit1"])
    ap.add_argument("--init", default="",
                    help="start from a saved factor checkpoint (save_factors format) instead "
                         "of building the arm, e.g. the bit1_qubo_seq init")
    ap.add_argument("--latent-scale", type=float, default=1.0,
                    help="rescale pre-sign latents on load; positive keeps every sign, "
                         "1/5 puts a full-rank arm on BitPath's sign-flip footing")
    ap.add_argument("--pool", default="puzzles", choices=["puzzles", "mixed"],
                    help="mixed = 80 %% real full-game positions + 20 %% puzzles")
    ap.add_argument("--no-game-eval", action="store_true",
                    help="skip the held-out full-game readout")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--lr_sign", type=float, default=1e-4)
    ap.add_argument("--lr_scale", type=float, default=2e-4)
    ap.add_argument("--lr_min", type=float, default=1e-5,
                    help="cosine floor; keeps the STE sign flips from firing late")
    ap.add_argument("--save-best", action="store_true",
                    help="also keep the epoch with the lowest held-out game KL")
    ap.add_argument("--no-sign", action="store_true",
                    help="rank/binarisation control: same U,V^T and rank, no sign()")
    ap.add_argument("--scale_max", type=float, default=4.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--w-hidden", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="write the trained factors to results/bt4/step2_weights_<tag>.pt")
    args = ap.parse_args()
    if args.prep:
        return prep(args)
    # joint_itq starts from a random orthogonal R, so the arm's sign pattern is
    # stochastic: seed it or two "itq" runs are different experiments.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    planes_path, meta_path = ((MIXED_PLANES, MIXED_META) if args.pool == "mixed"
                              else (PLANES, TRAIN))
    arr = np.load(planes_path, mmap_mode="r")
    N = arr.shape[0]
    rng = np.random.default_rng(0)
    train_rows = rows_of(meta_path)[:N]

    # teacher: untouched fp16 reference, resident, with a last-block tap
    teacher = _load_bt4(CKPT, "cpu").half().cuda().eval()
    tap = {}
    teacher.transformer.layers[-1].register_forward_hook(
        lambda m, i, o: tap.__setitem__("h", o.detach()))
    for p in teacher.parameters():
        p.requires_grad_(False)

    # student: quantise a second instance, then swap in the factored paths
    student = _load_bt4(CKPT, "cpu").half().cuda()
    cal_planes = None
    if args.init:
        # a saved init already carries the arm's signs, scales and latents, so
        # nothing is re-quantised: the run starts exactly where the gate ended
        student, _ih = load_factors(student, args.init, latent_scale=args.latent_scale)
        arm_name = "bit1_qubo_seq"
    elif args.arm == "bit1":
        # full rank, one sign per weight: no rank truncation at all
        _, parts = q1.quantise_bit1(student)
        bp.install_full(student, parts, sign=not args.no_sign)
        arm_name = "bit1"
    else:
        calib = [r for r in rows_of(PUZZLES) if r["split"] == "calib"]
        cal_planes = torch.stack([_leela_planes(live_board(r)) for r in calib]).cuda().half()
        _, _, paths = q1.quantise(student, cal_planes, args.arm, args.rounds)
        bp.install(student, paths, sign=not args.no_sign)
        arm_name = args.arm
    student = student.float().cuda()
    del cal_planes
    torch.cuda.empty_cache()
    if args.init:
        print("NOTE attention=explicit SDPA (install_full); the sequential init's capture "
              "patch was checked at 2.84e-3 relative logits", flush=True)

    stop = {}
    student.transformer.layers[-1].register_forward_hook(
        lambda m, i, o: stop.__setitem__("h", o))
    sign_p, scale_p = [], []
    for _, p in bp.bitpaths(student):
        sign_p += p.sign_params()
        scale_p += p.scale_params()
    # The factored arms' scales live below 1, so a fixed 4.0 bound was harmless
    # there. A full-rank path's row/column scales legitimately *start* above 4
    # (bit1's ALS fits |W| directly); clamping at 4 pins them from step 1 and
    # their gradient never lands, which is exactly how the first bit1 run stalled
    # at KL 2.46. Bound relative to initialisation instead.
    with torch.no_grad():
        smax0 = max(float(p.abs().max()) for p in scale_p)
    scale_max = max(args.scale_max, 4.0 * smax0)
    opt = torch.optim.AdamW([dict(params=sign_p, lr=args.lr_sign),
                             dict(params=scale_p, lr=args.lr_scale)], weight_decay=0.0)
    # Step 4 ran a *constant* LR and paid for it: game_kl 0.6204 (step 3600) ->
    # 0.7616 (4000) as the hidden loss went 1.05 -> 4.86, and the checkpoint on
    # disk was the bad one. Cosine to lr_min plus a best-epoch save makes the
    # last 200 steps unable to throw the run away.
    frac_min = max(args.lr_min / args.lr_sign, 1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: frac_min + (1 - frac_min) * 0.5 *
        (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    n_tr = sum(p.numel() for p in sign_p + scale_p)
    bits = sum(p.storage_bits() for _, p in bp.bitpaths(student))
    n_w = sum(p.d_out * p.d_in for _, p in bp.bitpaths(student))
    head = dict(arm=arm_name, scope="scope2", space=args.pool, sign=not args.no_sign,
                seed=args.seed, init=str(args.init) or None,
                latent_scale=args.latent_scale, attention="explicit SDPA (install_full)",
                trainable=n_tr, bits=bits,
                bpw=round(bits / n_w, 4), scale_init_max=round(smax0, 3),
                scale_clamp=round(scale_max, 3), batch=args.batch, steps=args.steps,
                lr_sign=args.lr_sign, lr_scale=args.lr_scale, pool=N,
                w=dict(policy=args.w_policy, value=args.w_value, hidden=args.w_hidden))
    print("HEAD", json.dumps(head), flush=True)

    ev = Eval("test")
    gev = None if args.no_game_eval else GameEval(teacher)
    masks = Masks(train_rows, ev.vocab)

    # Sign-flip accounting for a full-rank arm: `latent_scale` is only useful if
    # signs actually cross zero, so every readout reports the cumulative share of
    # signs that differ from the init, plus the churn since the last readout
    # (a flip that flips back shows up in frac but not in churn).
    full_rank = any(p.kind == "full" for _, p in bp.bitpaths(student))

    def scope_signs(model):
        return {nm: torch.where(p.B.detach() >= 0, 1.0, -1.0).to(torch.int8).cpu()
                for nm, p in bp.bitpaths(model) if p.kind == "full"}

    init_signs = scope_signs(student) if full_rank else {}
    prev_signs = {k: v.clone() for k, v in init_signs.items()}
    n_sign_bits = sum(v.numel() for v in init_signs.values())

    def flip_stats(obj, step):
        cur = scope_signs(obj)
        flipped = sum(int((cur[k] != init_signs[k]).sum()) for k in cur)
        churn = sum(int((cur[k] != prev_signs[k]).sum()) for k in cur)
        for k in cur:
            prev_signs[k].copy_(cur[k])
        st = dict(step=step, flip_frac=round(flipped / n_sign_bits, 6),
                  flip_churn=round(churn / n_sign_bits, 6), flipped=flipped,
                  n_bits=n_sign_bits)
        print("SIGNFLIP", json.dumps(st), flush=True)
        return dict(flip_frac=st["flip_frac"], flip_churn=st["flip_churn"])

    def readout(obj, step):
        rec = dict(step=step, **ev(obj))
        if gev is not None:
            rec.update(gev(obj))
        if full_rank:
            rec.update(flip_stats(obj, step))
        return rec

    tag = args.tag or f"{args.arm}_s{args.steps}"
    hist = [readout(student, 0)]
    print("EVAL", json.dumps(hist[-1]), flush=True)
    best = (float("inf"), 0, None)

    student.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, N, args.batch)
        planes = torch.from_numpy(np.asarray(arr[idx])).cuda()
        mask = masks.get(idx)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tpol, tval, _ = teacher(planes)
        thid = tap["h"].float()
        with torch.autocast("cuda", dtype=torch.float16):
            spol, sval, _ = student(planes)
        loss, parts = compute_loss(spol.float(), sval.float(), stop["h"].float(),
                                   tpol.float(), tval.float(), thid, mask, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(sign_p + scale_p, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():                     # keep the fp16 forward in range
            for p in scale_p:
                p.clamp_(1e-2, scale_max)
        if step % 25 == 0:
            smax = max(float(p.abs().max()) for p in scale_p)
            print(f"  step {step}  {json.dumps(parts)}  gnorm {float(gn):.3f}"
                  f"  scale_max {smax:.3f}  lr {opt.param_groups[0]['lr']:.2e}"
                  f"  {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % args.every == 0:
            torch.cuda.empty_cache()      # 8 GB card: the fp32 readout needs the room
            hist.append(readout(student, step))
            print("EVAL", json.dumps(hist[-1]), flush=True)
            key = hist[-1].get("game_kl", hist[-1]["kl"])
            if args.save_best and key < best[0]:
                best = (key, step, save_factors(student, f"{tag}_best", head))
                print("BEST", json.dumps(dict(step=step, game_kl=key,
                                              path=str(best[2]))), flush=True)
            student.train()

    rec = dict(head=head, history=hist, wall_s=round(time.time() - t0, 1),
               best=dict(step=best[1], game_kl=best[0] if best[2] else None,
                         path=str(best[2]) if best[2] else None),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"step2_qat_{tag}.json").write_text(json.dumps(rec, indent=1))
    if args.save:
        print("SAVED", save_factors(student, tag, head), flush=True)
    print("WALL", rec["wall_s"], "peak_gib", rec["peak_gib"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
