# Step 1 — 1.0 bpw on BT4 Scope 2: four initialisers, zero-shot

Scope 2 (`149.91M` = every 2-D weight in the 15 blocks: smolgen `sm1/sm2/sm3`,
packed `in_proj_weight`, `out_proj`, MLP `linear1/linear2`) at **1.0 bpw**.
Calibrated on the 512 calib positions, scored on the 2,000 test positions, all
against the cached FP16 reference. Raw: `results/bt4/step1_{quant,eval}_*.json`.

Measured bit cost: `1.0415 bpw` for the LittleBit arms (sign bits + per-row/col
fp16 scales), `1.0369 bpw` for the elementwise reference. 105 matrices.

## Results

| arm | solve@1 | solve@5 | top-1 agree | legal KL | last-block drift | bpw | build |
|---|---|---|---|---|---|---|---|
| **FP16 teacher** | **0.9440** | **0.9980** | — | — | 0.0 | 16 | — |
| `itq` | 0.1375 | 0.4290 | 0.1190 | 2.4711 | 1.7267 | 1.0415 | 111 s |
| `bit1` (full-rank 1-bit) | 0.0950 | 0.3070 | 0.0645 | 2.4728 | 1.5274 | 1.0369 | 0.6 s |
| `qubo_act` | **0.1915** | **0.4680** | **0.1690** | **2.3191** | **1.1901** | 1.0415 | 153 s |
| `qubo_seq` | 0.1255 | 0.3295 | 0.0965 | 2.5164 | **1.4616** | 1.0415 | 190 s |

Per-block drift, `‖stream_q − stream_fp16‖ / ‖stream_fp16‖` after each of the 15
blocks (`qubo_act` vs `qubo_seq`):

| block | 0 | 3 | 7 | 11 | 14 |
|---|---|---|---|---|---|
| `itq` | 1.39 | 2.12 | 2.73 | 2.48 | 1.73 |
| `bit1` | 1.92 | 1.86 | 4.02 | 3.77 | 1.53 |
| `qubo_act` | 0.79 | 1.22 | 1.39 | 1.33 | 1.19 |
| `qubo_seq` | 0.82 | 1.29 | 1.55 | 1.80 | 1.46 |

## Gate verdicts (frozen in STEP0.md §0.6, A1)

- **G1 — FAILS, and in the direction opposite to the hypothesis.** `qubo_seq`
  ends at 1.46 last-block drift against `qubo_act`'s 1.19, and it is worse at
  every one of the 15 blocks. Sequential error compensation does
  **not** cancel cross-layer drift on a 15-layer Chessformer; it accumulates it.
- **G2 — FAILS.** Best arm agrees with the FP16 top-1 move 16.9 % of the time
  (gate: not graded, but the abort floor of 5 % was cleared, so the wiring is
  sound).
- **G3 — FAILS.** Legal-move KL 2.32 nats for the best arm. For scale, the FP16
  policy's own mean entropy is 0.80 nats: the quantised policy is not a
  perturbation of the teacher, it is a different, nearly uniform distribution.
- **G4 — FAILS hard.** 0.19 solve@1 against a 0.66 gate; 0.47 solve@5 against a
  0.90 gate. Zero-shot 1.0 bpw is not a functioning reranker of moves.

**Nothing aborts.** Per the request, the sharp zero-shot drop is the expected
result of post-training binarisation, and Step 2 (policy/value distillation) is
the measurement that decides whether 1.0 bpw is viable at all — exactly as it was
for Qwen3, where the init gate also read chance and only KD moved the number.

## What the arms actually tell us

1. **The rank-bottleneck hypothesis is falsified.** I added `bit1` (full-rank
   elementwise `h_i·sign(W_ij)·g_j`, no low-rank path, same 1.0 bpw) expecting
   the rank-`d/2` LittleBit path's `sqrt(1−r/d)` floor to be the cause of the
   collapse. It is not: the full-rank scheme is **worse** end to end (9.5 % vs
   13.8 % solve@1). The LittleBit path's ITQ rotation plus per-rank scale buys
   more than the lost rank costs. Binarisation, not the bottleneck, is what
   destroys the model at this bit rate.

2. **QUBO helps, measurably, wherever the weights have structure.** With
   activation-weighted Σ and one Jacobi column pass, the MLP layers drop a lot —
   e.g. `blk14.ff2` 0.5547 → 0.2130 and `blk13.ff2` 0.4371 → 0.2326 in
   activation-normalised error — while `sm1` (32×1024, rank 31) barely moves
   (0.6553 → 0.5559). End to end that is worth +5.4 pp solve@1 and +5.0 pp
   agreement over `itq`, and it cuts last-block drift from 1.73 to 1.19. So the
   solver is working; the objective is just far too weak a proxy at this bit
   rate.

3. **`qubo_seq` optimises the wrong distribution.** Per layer, measured on the
   drifted stream it was fitted to, its refinement looks *better* than
   `qubo_act`'s: `blk14.ff2` 0.5760 → **0.0975** vs 0.5547 → 0.2130, `blk11.sm1`
   0.6829 → **0.3656** vs 0.6390 → 0.5792. That is the trap. Each layer is fitted
   to an input manifold that the next quantisation moves again, so the layers'
   objectives are mutually inconsistent; the local error is smaller and the
   end-to-end result is worse on every metric. The drift trace makes the
   mechanism visible: `qubo_seq` *starts* level with `qubo_act` (0.82 vs 0.79 at
   block 0) and falls behind monotonically from block 3 (1.29 vs 1.22) to block
   11 (1.80 vs 1.33).

## Caveats

- One calibration set, one seed, one bit rate; no repeats, so differences below
  ~1 pp should not be read as real.
- `out_proj`'s Σ is an **approximation**: PyTorch's fused
  `F.multi_head_attention_forward` never calls the `out_proj` module, so its hook
  cannot fire and its calibration uses the attention *input* second moment (same
  1024 dims). 15 of 105 matrices are affected; the other 90 capture exactly.
- The build is `rounds=1` with the whole-column (Jacobi) solve rather than 2
  rounds with block-Gauss-Seidel: GS cost 4 s on the smallest matrix and stalled
  on the 2048-wide ones. Both are the same column QUBO; GS is a stronger
  optimiser, so `qubo_*` numbers here are a **lower bound** on what the solver
  can do.
- `solve@1/@5` are the strict 1-node puzzle readouts; they are not chess strength
  (that is Step 3's match), and a 1-bit model at 19 % is not evidence about
  strength at any search budget.
- The puzzle set is tactical only (STEP0.md §0.4).

## Files

`harness/bt4/q1bit.py` → `results/bt4/step1_quant_{itq,bit1,qubo_act,qubo_seq}.json`,
`results/bt4/step1_eval_*.json`, logs `results/bt4/step1_qubo_{act,seq}_log.txt`.
Weights are quantised in place as the exact dense equivalent, so the captured
activations are the real quantised stream.
