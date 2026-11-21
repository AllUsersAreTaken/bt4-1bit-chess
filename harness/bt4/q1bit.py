"""Step 1: 1.0 bpw binarisation of BT4 Scope 2, three initialisers.

Scope 2 = every 2-D weight inside the 15 blocks: smolgen sm1/sm2/sm3, the packed
QKV `mha.in_proj_weight`, `mha.out_proj`, and the MLP `linear1`/`linear2`.
That is 149.91M weights (81 % of the 185.25M the model uses); the input block and
the heads stay fp16 so this isolates whether the Transformer body survives 1 bit.

1.0 bpw = one LittleBit path of rank r per matrix with

    r = round(d_in * d_out / (d_in + d_out))

since a rank-r path stores (d_in + d_out) * r sign bits for d_in * d_out weights.

Arms
  itq       inverse-transform init, no QUBO (LittleBit-2's baseline)
  qubo_act  itq, then row/column QUBO refinement against Sigma from the *clean*
            fp16 activations
  qubo_seq  the same refinement but layer by layer in execution order, with
            Sigma taken from the activations the already-quantised model emits,
            so each layer sees its predecessor's binary drift

Weights are quantised into the module in place (W <- effective_weight()), which is
the exact dense equivalent of the binary forward, so the captured activations are
the real quantised stream.

usage:
  python harness/bt4/q1bit.py --arm itq
  python harness/bt4/q1bit.py --arm qubo_act --rounds 2
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time
import types

import chess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "vendor/little-bit")
from littlebit.layer import _Path                                     # noqa: E402

from chessformer_lens.adapters.lc0 import (_leela_all_moves, _leela_planes,
                                           _load_bt4, Lc0Vocab)       # noqa: E402

_spec = importlib.util.spec_from_file_location("lbq", "harness/littlebit/qubo.py")
lbq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lbq)

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")
CACHE = pathlib.Path("results/bt4/_cache")
OUT = pathlib.Path("results/bt4")


# ----------------------------------------------------------------- the targets
class Target:
    """One 2-D weight, quantised in place, with the module to hook for Sigma."""

    def __init__(self, label, module, get_w, set_w, hook):
        self.label, self.module, self.get_w, self.set_w, self.hook = \
            label, module, get_w, set_w, hook
        w = get_w()
        self.shape = tuple(w.shape)
        self.d_out, self.d_in = self.shape
        self.r = max(1, round(self.d_out * self.d_in / (self.d_out + self.d_in)))
        self.bits = (self.d_out + self.d_in) * self.r + 16 * (self.d_out + self.d_in + self.r)

    def __repr__(self):
        return f"{self.label} {self.shape} r={self.r}"


def targets_of(model) -> list[Target]:
    """Execution order inside a block: smolgen bias first, then attention, then MLP."""
    out = []
    for i, blk in enumerate(model.transformer.layers):
        attn = blk.self_attn
        p = f"blk{i:02d}"
        for nm in ("sm1", "sm2", "sm3"):
            lin = getattr(attn, nm)
            out.append(Target(f"{p}.{nm}", lin,
                              lambda l=lin: l.weight.data, lambda w, l=lin: setattr(l.weight, "data", w),
                              lin))
        mha = attn.mha
        out.append(Target(f"{p}.in_proj", mha,
                          lambda m=mha: m.in_proj_weight.data,
                          lambda w, m=mha: setattr(m, "in_proj_weight", nn.Parameter(w.clone(), requires_grad=False)),
                          mha))
        # out_proj is consumed by the fused F.multi_head_attention_forward, which
        # never calls the module, so its hook cannot fire: calibrate it on the
        # attention input's second moment (same 1024 dims) instead. The head
        # outputs are a convex mix of the V projections, so this is an
        # approximation, documented in STEP1.md, not an exact capture.
        out.append(Target(f"{p}.out_proj", mha.out_proj,
                          lambda l=mha.out_proj: l.weight.data,
                          lambda w, l=mha.out_proj: setattr(l.weight, "data", w), mha))
        out.append(Target(f"{p}.ff1", blk.linear1,
                          lambda l=blk.linear1: l.weight.data,
                          lambda w, l=blk.linear1: setattr(l.weight, "data", w), blk.linear1))
        out.append(Target(f"{p}.ff2", blk.linear2,
                          lambda l=blk.linear2: l.weight.data,
                          lambda w, l=blk.linear2: setattr(l.weight, "data", w), blk.linear2))
    return out


# --------------------------------------------------------------- calibration
class SigmaCatcher:
    """Accumulates Sigma = X^T X (float32) for one module."""

    def __init__(self, module, d_in, max_rows=None):
        self.d_in, self.sigma, self.n = d_in, None, 0
        self.h = module.register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        x = args[0].detach().float()
        X = x.reshape(-1, self.d_in)
        if self.sigma is None:
            self.sigma = torch.zeros(self.d_in, self.d_in, device=X.device, dtype=torch.float32)
        step = 8192
        for s in range(0, X.size(0), step):
            c = X[s:s + step]
            self.sigma += c.mT @ c
        self.n += X.size(0)

    def close(self):
        self.h.remove()
        return None if self.sigma is None else (self.sigma / max(self.n, 1)).cpu()


def run(model, planes, batch=64):
    with torch.no_grad():
        for s in range(0, planes.size(0), batch):
            model(planes[s:s + batch])


def quantise_bit1(model, verbose=True):
    """Full-rank elementwise 1-bit reference: W ~ h_i * sign(W_ij) * g_j.

    Exactly 1.0 bpw (one sign per weight + per-row/col fp16 scales) and no rank
    bottleneck, which is the point: it separates "binarisation costs accuracy"
    from "a rank-d/2 path costs accuracy", the latter having a sqrt(1 - r/d)
    error floor that has nothing to do with 1-bit arithmetic.
    """
    recs, parts = [], {}
    for t in targets_of(model):
        w = t.get_w().detach().float()
        s = torch.where(w >= 0, 1.0, -1.0)
        a = w * s                                        # |w| with the sign removed
        g = torch.ones(w.size(1), device=w.device)
        for _ in range(4):                               # alternating least squares
            h = (a @ g) / g.pow(2).sum()
            g = (a.mT @ h) / h.pow(2).sum().clamp_min(1e-12)
        wq = h[:, None] * s * g[None, :]
        base = float(w.pow(2).sum().sqrt())
        err = float((w - wq).pow(2).sum().sqrt()) / base
        t.set_w(wq.to(torch.float16))
        # the factors Step 2's QAT has to start from: full-rank signs + scales
        parts[t.label] = types.SimpleNamespace(S=s.cpu().clone(), h=h.cpu().clone(),
                                               g=g.cpu().clone())
        recs.append(dict(label=t.label, shape=list(t.shape), r="full",
                         bits=t.d_out * t.d_in + 16 * (t.d_out + t.d_in),
                         act_err_after=round(err, 6)))
        if verbose and "ff2" in t.label:
            print(f"  [bit1] {t.label:14s} {str(t.shape):14s} frob err {err:.4f}", flush=True)
    return recs, parts


def quantise(model, planes, arm, rounds, cols="jacobi", verbose=True):
    """Build the arm in place. Returns per-target weight-error records.

    Also returns {label: _Path} — the binary factors themselves, which Step 2's
    QAT needs as its initialisation; the in-place dense write is only the
    forward-equivalent of exactly these factors.
    """
    tg = targets_of(model)
    W0 = [t.get_w().detach().float().clone().cpu() for t in tg]
    recs, paths = [], {}

    if arm == "qubo_act":
        cats = [SigmaCatcher(t.hook, t.d_in) for t in tg]
        run(model, planes)                      # one pass over the clean fp16 model
        sig = {t.label: c.close() for t, c in zip(tg, cats)}
        if verbose:
            print(f"  captured {len(sig)} clean-activation Sigmas in one pass", flush=True)

    done = set()
    for k, t in enumerate(tg):
        w = W0[k].to("cuda")
        t0 = time.time()
        path = _Path(w, t.r, 100.0, "itq")
        if arm == "itq":
            sigma = None
        elif arm == "qubo_act":
            sigma = sig[t.label].cuda()
        else:                                   # qubo_seq: Sigma from the quantised stream
            c = SigmaCatcher(t.hook, t.d_in)
            run(model, planes)
            sigma = c.sigma.cuda()
            c.close()
        e_itq = lbq.path_error(w, path.effective_weight().to(w.dtype), sigma)
        if arm != "itq" and rounds:
            for _ in range(rounds):
                lbq.update_rows(path, w, sigma, "descent", 200, None, None, 4)
                if cols == "gs":
                    lbq.update_cols_gs(path, w, sigma, block=8, solver="descent",
                                       sa_sweeps=200, gen=None, restarts=4)
                else:                      # whole-column (Jacobi) QUBO, one batch
                    lbq.update_cols(path, w, sigma, cols=None, solver="descent",
                                    sa_sweeps=200, gen=None, restarts=4)
        paths[t.label] = path
        wq = path.effective_weight().detach().float()
        e_after = lbq.path_error(w, wq, sigma)
        base = lbq.path_error(w, torch.zeros_like(w), sigma) if sigma is not None else \
            float(w.pow(2).sum().sqrt())
        t.set_w(wq.to(torch.float16))
        done.add(t.label)
        recs.append(dict(label=t.label, shape=list(t.shape), r=t.r, bits=t.bits,
                         act_err_before=round(e_itq / base, 6),
                         act_err_after=round(e_after / base, 6),
                         secs=round(time.time() - t0, 2)))
        if verbose and (k % 7 == 0 or "ff2" in t.label):
            print(f"  [{arm}] {t.label:14s} {str(t.shape):14s} r={t.r:4d} "
                  f"err {e_itq/base:.4f} -> {e_after/base:.4f}  {recs[-1]['secs']}s", flush=True)
    return tg, recs, paths


# ------------------------------------------------------------------ evaluation
def capture_streams(model, planes, n_layers):
    """Post-block residual stream for every layer, float32, on the GPU."""
    outs, hooks = {}, []
    for i, blk in enumerate(model.transformer.layers):
        hooks.append(blk.register_forward_hook(
            lambda m, inp, out, i=i: outs.__setitem__(i, out.detach().float())))
    run(model, planes)
    for h in hooks:
        h.remove()
    return [outs[i] for i in range(n_layers)]


def drift_vs(streams_ref, streams_q):
    """Relative error of the quantised stream after each block."""
    out = []
    for a, b in zip(streams_ref, streams_q):
        num = (b - a).norm().item()
        den = a.norm().item()
        out.append(round(num / max(den, 1e-12), 6))
    return out


def evaluate(model, split, tag, streams_ref=None, drift_planes=None):
    rows = [json.loads(l) for l in PUZZLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    vocab = Lc0Vocab(_leela_all_moves())
    boards, gold = [], []
    for r in rows:
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        boards.append(b)
        gold.append(vocab.index(b, chess.Move.from_uci(r["moves"][1])))
    planes = torch.stack([_leela_planes(b) for b in boards]).cuda().half()
    masks = torch.stack([vocab.legal_mask(b) for b in boards]).cuda()
    gold_t = torch.tensor(gold, device="cuda")

    ref = np.load(CACHE / f"policy_fp16_{split}.npy").astype(np.float32) \
        if (CACHE / f"policy_fp16_{split}.npy").exists() else None

    n = len(rows)
    top1 = np.zeros(n, dtype=np.int64)
    logits = np.zeros((n, len(vocab.moves)), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, 64):
            pol, _, _ = model(planes[s:s + 64])
            logits[s:s + 64] = pol.float().cpu().numpy().astype(np.float16)
            top1[s:s + 64] = pol.float().masked_fill(~masks[s:s + 64], float("-inf")).argmax(-1).cpu().numpy()
    infer_s = time.time() - t0

    m = np.where(masks.cpu().numpy(), logits.astype(np.float32), -np.inf)
    rank = (m > m[np.arange(n), np.array(gold)][:, None]).sum(1)
    rec = dict(
        split=split, n=n, infer_s=round(infer_s, 2),
        pos_per_s=round(n / infer_s, 1),
        solve_top1=round(float((rank == 0).mean()), 4),
        solve_top5=round(float((rank < 5).mean()), 4),
    )
    if ref is not None:
        legal = masks.cpu().numpy()
        lp_ref = torch.log_softmax(torch.tensor(ref).masked_fill(~torch.tensor(legal), float("-inf")), -1)
        lp_q = torch.log_softmax(torch.tensor(logits.astype(np.float32)).masked_fill(~torch.tensor(legal), float("-inf")), -1)
        kl = torch.where(torch.tensor(legal), lp_ref.exp() * (lp_ref - lp_q), torch.zeros(1)).sum(-1)
        agree = (np.array(top1) == np.array(ref.argmax(1))).mean()
        rec.update(policy_top1_agreement=round(float(agree), 4),
                   legal_kl=round(float(kl.mean()), 4),
                   kl_median=round(float(kl.median()), 4))

    if streams_ref is not None and drift_planes is not None:
        drift = drift_vs(streams_ref, capture_streams(model, drift_planes, len(model.transformer.layers)))
        rec["block_drift"] = drift
        rec["last_block_drift"] = drift[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"step1_eval_{tag}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["itq", "qubo_act", "qubo_seq", "bit1"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cols", default="jacobi", choices=["jacobi", "gs"])
    ap.add_argument("--calib", default="calib")
    ap.add_argument("--eval", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in PUZZLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    cal = [r for r in rows if r["split"] == args.calib]
    boards = []
    for r in cal:
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        boards.append(b)
    planes = torch.stack([_leela_planes(b) for b in boards]).cuda().half()
    del boards

    model = _load_bt4(CKPT, "cpu").half().cuda().eval()

    # fp16 reference streams on a slice of the eval split, captured before any
    # weight changes, so G1 is measured against the real teacher stream.
    ev = [r for r in rows if r["split"] == args.eval][:128]
    drift_planes = torch.stack([_leela_planes(chess.Board(r["fen"])) for r in ev]).cuda().half()
    streams_ref = capture_streams(model, drift_planes, len(model.transformer.layers))
    del ev

    t0 = time.time()
    if args.arm == "bit1":
        recs, _ = quantise_bit1(model)
        tg = targets_of(model)
    else:
        tg, recs, _ = quantise(model, planes, args.arm, args.rounds, args.cols)
    quant_s = time.time() - t0
    total_bits = sum(int(r["bits"]) for r in recs)
    total_w = sum(t.d_out * t.d_in for t in tg)
    head = dict(arm=args.arm, scope="scope2", rounds=args.rounds,
                n_matrices=len(tg), quant_params=total_w, bits=total_bits,
                bpw=round(total_bits / total_w, 4), quant_s=round(quant_s, 1))
    tag = args.tag or f"{args.arm}"
    (OUT / f"step1_quant_{tag}.json").write_text(json.dumps(dict(head=head, per_layer=recs), indent=1))
    print("QUANT", json.dumps(head), flush=True)

    rec = evaluate(model, args.eval, tag, streams_ref, drift_planes)
    print("EVAL", json.dumps({k: v for k, v in rec.items() if k != "block_drift"}))
    print("drift", rec["block_drift"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
