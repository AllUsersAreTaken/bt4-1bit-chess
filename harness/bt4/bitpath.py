"""Step 2 core: the factored 1-bit path as a trainable module.

Step 1 quantised BT4 by writing `path.effective_weight()` into each module, so
its activations are the real binary stream but the factors are gone. QAT needs
the factors back, because the things that must move are `sign(U)`, `sign(V)`
and the three scales — and rebuilding 105 dense effective weights from
rank-r factors each step costs ~150 TFLOP, which is not a training loop.

So `BitPath` runs the factorisation forwards:

    y = ((x . g) @ sgn(V)) . l @ sgn(U)^T . h        W = b @ a^T

with `sgn` a straight-through estimator (identity on the backward pass). The
storage is unchanged: one sign bit per latent plus fp16 scales, i.e. exactly the
bit budget Step 1 measured.

`patch_attention` replaces nn.MultiheadAttention's fused call with an explicit
`scaled_dot_product_attention` so that the *packed* QKV projection and
`out_proj` both run as factored paths. The fused kernel reads
`in_proj_weight` directly, which is the only reason Step 1 had to materialise
dense weights (and why the `out_proj` hook never fired).

usage:
  python harness/bt4/bitpath.py --arm itq          # forward-equivalence test
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _leela_planes, _load_bt4   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
PUZZLES = pathlib.Path("data/bt4/puzzles.jsonl")
OUT = pathlib.Path("results/bt4")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


q1 = _load("q1bit", "harness/bt4/q1bit.py")


# ------------------------------------------------------------------ the module
class _Sgn(torch.autograd.Function):
    """sign(x) forwards; identity gradient (standard binary-net STE).

    LittleBit's own SmoothSign narrows the gradient to k(1-tanh(kx)^2); at the
    k=100 used for initialisation that is ~0 for every latent that is not
    already at zero, which would freeze the signs. Starting from a *trained*
    LittleBit initialiser, the plain STE is what lets a sign flip happen.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


def sgn(x):
    return _Sgn.apply(x)


class BitPath(nn.Module):
    """diag(h) . sign(U) . diag(l) . sign(V)^T . diag(g), factored forwards."""

    def __init__(self, path, sign=True, binarise_init=None):
        """`sign=False` keeps the factors continuous (the binarisation control).

        `binarise_init` decides what a continuous path starts from: the arm's
        pre-sign latents are *not* a good factorisation on their own, so a fresh
        control starts from the binary weight's own +-1 factors (same W, same
        rank, no constraint). A checkpoint that is already continuous passes
        `binarise_init=False` and is loaded verbatim.
        """
        super().__init__()
        if binarise_init is None:
            binarise_init = not sign
        self.signed = bool(sign)
        # The arm stores the *pre-sign* continuous latents, so dropping sign()
        # on them would start the control from a different (and much worse)
        # weight matrix. The control must start from the binary weight's own
        # +-1 factors as continuous numbers: same W, same rank, no constraint.
        self.U = nn.Parameter(path.U.detach().sign().clone() if binarise_init
                              else path.U.detach().clone())
        self.V = nn.Parameter(path.V.detach().sign().clone() if binarise_init
                              else path.V.detach().clone())
        self.h = nn.Parameter(path.h.detach().clone())
        self.g = nn.Parameter(path.g.detach().clone())
        self.l = nn.Parameter(path.l.detach().clone())
        self.d_out, self.r = self.U.shape
        self.d_in = self.V.shape[0]

    def forward(self, x):
        V = sgn(self.V) if self.signed else self.V
        U = sgn(self.U) if self.signed else self.U
        a = V * self.g[:, None]                                    # (d_in, r)
        b = U * self.h[:, None] * self.l[None, :]                  # (d_out, r)
        return (x @ a) @ b.mT                                      # (..., d_out)

    @torch.no_grad()
    def effective_weight(self):
        V = torch.where(self.V >= 0, 1.0, -1.0) if self.signed else self.V
        U = torch.where(self.U >= 0, 1.0, -1.0) if self.signed else self.U
        a = V * self.g[:, None]
        b = U * self.h[:, None] * self.l[None, :]
        return b @ a.mT                                            # (d_out, d_in)

    def storage_bits(self):
        latent = (self.d_in + self.d_out) * self.r
        return (latent if self.signed else 16 * latent) + 16 * (self.d_in + self.d_out + self.r)

    kind = "factored"

    def sign_params(self):
        return [self.U, self.V]

    def scale_params(self):
        return [self.h, self.g, self.l]


class BitFull(nn.Module):
    """`h . sign(B) . g` — full rank, one sign bit per weight, no rank tax.

    Scope-2 matrices of BT4 are nowhere near rank d/2 (exact SVD at that rank
    loses 19.5 % of each matrix and takes solve@1 from 0.9440 to 0.3110), so the
    littlebit path was paying a rank tax on top of the bit. This is the same
    1.0 bpw spent as one sign per weight instead, which has no rank bottleneck.
    """

    kind = "full"

    def __init__(self, part, sign=True, latent_init="sign", latent_scale=1.0):
        """`latent_init` picks what the sign latent starts at.

        "sign" is the original `part.S`: the +-1 pattern itself, so |B| = 1.0
        exactly and Adam (step ~lr) cannot carry a latent across zero -- the
        signs are frozen for the whole run. "ratio" keeps the same signs but
        stores the pre-sign magnitude `W/(h g)`, whose small entries sit near 0
        and can flip early. The forward pass uses sgn(B), so the effective
        weight is identical either way; only the training dynamics differ.
        """
        super().__init__()
        self.signed = bool(sign)
        lat = getattr(part, "latent", None) if latent_init == "ratio" else None
        if lat is not None:
            # A positive rescale cannot change sgn(), so the model is identical;
            # it only moves every latent closer to (or further from) the 0 that a
            # sign flip has to cross. BitPath's ITQ latents sit at |.| ~ 0.05, so
            # scale ~1/5 puts a full-rank latent on the same flip footing.
            lat = lat * float(latent_scale)
        self.B = nn.Parameter((lat if lat is not None else part.S).detach().clone())
        self.h = nn.Parameter(part.h.detach().clone())
        self.g = nn.Parameter(part.g.detach().clone())
        self.d_out, self.d_in = self.B.shape

    def forward(self, x):
        B = sgn(self.B) if self.signed else self.B
        return ((x * self.g) @ B.mT) * self.h

    @torch.no_grad()
    def effective_weight(self):
        B = torch.where(self.B >= 0, 1.0, -1.0) if self.signed else self.B
        return self.h[:, None] * B * self.g[None, :]

    def storage_bits(self):
        n = self.d_out * self.d_in
        return (n if self.signed else 16 * n) + 16 * (self.d_out + self.d_in)

    def sign_params(self):
        return [self.B]

    def scale_params(self):
        return [self.h, self.g]


class PathLinear(nn.Module):
    """nn.Linear replacement: factored path + the original (fp16) bias."""

    def __init__(self, bp, bias=None):
        super().__init__()
        self.bp = bp
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone())

    def forward(self, x):
        y = self.bp(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)


def _sdpa_forward(self, x):
    """SmolgenAttention.forward, with both projections factored."""
    B, T, d = x.shape
    H, hd = self.num_heads, d // self.num_heads
    bias = self._sq_bias(x)                                    # (B, H, T, T)
    qkv = self.in_path(x)
    b_in = self.mha.in_proj_bias                      # packed [q; k; v], (3d,)
    if b_in is not None:
        qkv = qkv + b_in.to(qkv.dtype)
    q, k, v = qkv.split(d, dim=-1)

    def heads(t):
        return t.reshape(B, T, H, hd).transpose(1, 2)

    y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=bias)
    y = y.transpose(1, 2).reshape(B, T, d)
    out = self.out_path(y)
    b = self.mha.out_proj.bias
    return (out if b is None else out + b.to(out.dtype)), None


def patch_attention(attn, in_bp, out_bp):
    attn.in_path = in_bp
    attn.out_path = out_bp
    attn.forward = types.MethodType(_sdpa_forward, attn)


def install(model, paths, sign=True, binarise_init=None):
    """Swap every Scope-2 module for its factored path, in place.

    `paths` maps the Step 1 labels to the `_Path` the arm produced, so the
    student starts *exactly* at the arm's quantised weights. `sign=False` keeps
    the same factors and rank but leaves them continuous (the binarisation
    control).
    """
    def bp_(label):
        return BitPath(paths[label], sign=sign, binarise_init=binarise_init)

    for i, blk in enumerate(model.transformer.layers):
        p, attn = f"blk{i:02d}", blk.self_attn
        for nm in ("sm1", "sm2", "sm3"):
            lin = getattr(attn, nm)
            setattr(attn, nm, PathLinear(bp_(f"{p}.{nm}"), lin.bias))
        patch_attention(attn, bp_(f"{p}.in_proj"), bp_(f"{p}.out_proj"))
        blk.linear1 = PathLinear(bp_(f"{p}.ff1"), blk.linear1.bias)
        blk.linear2 = PathLinear(bp_(f"{p}.ff2"), blk.linear2.bias)
    return model


def install_full(model, parts, sign=True, latent_init="sign", latent_scale=1.0):
    """Same swap as `install`, but for the full-rank `BitFull` factors."""
    def m_(label):
        return BitFull(parts[label], sign=sign, latent_init=latent_init,
                       latent_scale=latent_scale)

    for i, blk in enumerate(model.transformer.layers):
        p, attn = f"blk{i:02d}", blk.self_attn
        for nm in ("sm1", "sm2", "sm3"):
            lin = getattr(attn, nm)
            setattr(attn, nm, PathLinear(m_(f"{p}.{nm}"), lin.bias))
        patch_attention(attn, m_(f"{p}.in_proj"), m_(f"{p}.out_proj"))
        blk.linear1 = PathLinear(m_(f"{p}.ff1"), blk.linear1.bias)
        blk.linear2 = PathLinear(m_(f"{p}.ff2"), blk.linear2.bias)
    return model


def bitpaths(model):
    """(label, BitPath) for every installed path, for the optimiser + bit count."""
    out = []
    for i, blk in enumerate(model.transformer.layers):
        p, attn = f"blk{i:02d}", blk.self_attn
        for nm in ("sm1", "sm2", "sm3"):
            out.append((f"{p}.{nm}", getattr(attn, nm).bp))
        out += [(f"{p}.in_proj", attn.in_path), (f"{p}.out_proj", attn.out_path)]
        out += [(f"{p}.ff1", blk.linear1.bp), (f"{p}.ff2", blk.linear2.bp)]
    return out


# ------------------------------------------------------------------ the check
def calib_planes(n=8, rows=None):
    rows = rows or [json.loads(l) for l in PUZZLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == "calib"][:n]
    boards = []
    for r in rows:
        b = chess.Board(r["fen"])
        b.push(chess.Move.from_uci(r["first"]))
        boards.append(b)
    return torch.stack([_leela_planes(b) for b in boards])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="itq", choices=["itq", "qubo_act", "qubo_seq"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    planes = calib_planes(args.n).cuda().float()
    model = _load_bt4(CKPT, "cpu").float().cuda().eval()

    tg, recs, paths = q1.quantise(model, planes, args.arm, args.rounds)
    # quantise() writes fp16 weights (right for Step 1's inference readout, but
    # the equivalence check has to compare two float32 functions: the dense
    # write of the arm's effective weight vs the factored module built from the
    # very same factors).
    for t in tg:
        t.set_w(paths[t.label].effective_weight().float())
    model = model.float()
    with torch.no_grad():
        dense_pol, dense_val, _ = model(planes)

    model = install(model, paths).eval()
    with torch.no_grad():
        path_pol, path_val, _ = model(planes)

    def rel(a, b):
        return float((a - b).abs().max()) / max(float(a.abs().max()), 1e-12)

    d_pol = float((dense_pol - path_pol).abs().max())
    d_val = float((dense_val - path_val).abs().max())
    scale = float(dense_pol.abs().max())
    W = model.transformer.layers[0].self_attn.in_path.effective_weight()
    Wd = paths["blk00.in_proj"].effective_weight()
    d_w = float((W - Wd.float()).abs().max())

    # fp16 throughput of the factored student (the training-loop reality)
    model = model.half()
    p16 = planes.half()
    with torch.no_grad():
        model(p16[:1])
        torch.cuda.synchronize()
        t0 = __import__("time").time()
        for _ in range(3):
            model(p16)
        torch.cuda.synchronize()
        dt = (__import__("time").time() - t0) / 3
    pos_s = args.n / dt
    bits = sum(b.storage_bits() for _, b in bitpaths(model))
    n_w = sum(b.d_out * b.d_in for _, b in bitpaths(model))

    rec = dict(arm=args.arm, n_positions=args.n, logit_scale=round(scale, 3),
               max_abs_delta_policy=round(d_pol, 9),
               max_abs_delta_value=round(d_val, 9),
               max_rel_delta_policy=round(d_pol / max(scale, 1e-12), 9),
               max_abs_delta_inproj_weight=round(d_w, 9),
               inproj_rel=round(d_w / max(float(Wd.abs().max()), 1e-12), 9),
               bits=bits, n_weights=n_w, bpw=round(bits / n_w, 4),
               fp16_pos_per_s=round(pos_s, 1),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               pass_dense_vs_path=bool(d_pol / max(scale, 1e-12) < 1e-4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step2_bitpath_check.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0 if rec["pass_dense_vs_path"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
