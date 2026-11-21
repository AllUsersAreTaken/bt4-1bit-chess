# Step 0 — BT4 feasibility, parameter audit, FP16 baseline

All numbers below were produced by the scripts in `harness/bt4/` on this machine
(RTX 4060 8 GB, torch 2.11.0+cu128) and are read back from
`results/bt4/*.json`. Nothing here is estimated or copied from a model card.

> **Amendment A1 (2026-11-21).** The first version of this file reported
> `solve@1 = 6.30 %` and read the Lichess move list one ply too early. Per
> database.lichess.org, "FEN is the position before the opponent makes their
> move… The second move is the beginning of the solution", so `Moves[0]` is the
> *opponent's* blunder and `Moves[1]` is the solver's move. The ingest now pushes
> `Moves[0]` and grades on `Moves[1]`. Corrected reference: see §0.5. The
> consequence for the frozen gate is recorded in §0.6. This was caught before any
> quantised arm was scored.

## 0.1 What was pulled and how it runs

| item | value |
|---|---|
| source | `LeelaChessZero/BT4` → `BT4-1024x15x32h-swa-6147500.pb.gz` |
| file | 335,897,546 B, sha256 `66e6c1b5…e545c67e` (matches lc0's published hash) |
| decode | pure-PyTorch BT4 from `chessformer_lens` 3.2 `adapters/lc0.py` — no ONNX runtime, no lc0 binary, no protobuf compiler |
| load time | 2.8 s (CPU parse + build), 0.74 GiB VRAM at batch 64 fp16 |
| forward | `planes (B,112,8,8) → policy (B,4288), WDL (B,3), moves-left (B,)` |

Correctness checks before trusting anything downstream:

- Start position top-5 policy: `d2d4, g1f3, c2c4, g2g3, e2e3`; WDL `0.19 / 0.64 / 0.17`.
- Legal-move vocabulary coverage **100 %**: 46/46, 27/27, 20/20, 5/5, 7/7 on
  positions including castling-rights and promotion cases. `Lc0Vocab` handles
  lc0's quirks (castling as king-takes-rook, knight promotions on the plain
  square pair), so a masked argmax is always a real legal move.
- Tactical recognition spot checks: finds `b3b8+` (Morphy's Opera Game queen
  sacrifice), `f1b5` in the Ruy Lopez, `a7a8=Q` in a promotion race.
- Puzzle move convention verified against the published format and against the
  data itself: every puzzle row has an **even** move count (4, 6, 8, 2 …) and all
  182 `mateIn1` puzzles have exactly 2 moves — blunder, then mate.

## 0.2 Parameter audit — the plan's premise does not hold

The file carries **206.0M** parameters, not 191M, in 463 tensors. It is also not
"nearly all `nn.Linear` inside the 15 blocks". Measured split of the file:

| category | params | share |
|---|---|---|
| attention q/k/v/out (15 × 4 × 1024²) | 62.91M | 30.5 % |
| **smolgen** (15 × 3.73M: compress, board-summary dense, gen_from 8192×256) | 55.92M | 27.2 % |
| MLP fc1/fc2 (15 × 2 × 1024×1536) | 47.19M | 22.9 % |
| **input_block** (preprocess 32768×768 = 25.17M, embed, gating, FFN) | 29.12M | 14.1 % |
| policy heads (`policy_vanilla`, `policy_optimistic_st`, `policy_next`, `policy_opponent`, `policy_soft`, `policy_embed`) | 6.68M | 3.2 % |
| value / moves-left heads (`value_q`, `value_st`, `value_winner`, `mlh`) | 4.29M | 2.1 % |
| LayerNorms, biases, gates (all 1-D) | 0.20M | 0.1 % |

Only **185.25M / 349 tensors are used at inference**. The other 20.7M is dead
weight in the checkpoint:

- 15 redundant copies of the shared GAB template bank (`4096×256`), 14 of them
  never read — the bank is shared across layers and heads, so 14 × 1.049M ≈
  14.7M is duplicated;
- training-only heads this inference path never reaches: `policy_optimistic_st`
  (2.10M), `policy_next`/`policy_opponent`/`policy_soft` (1.58M), `value_q`,
  `value_st`, `value_winner` (1.73M) ≈ 5.4M.

So the honest denominator for a bit budget is **185.25M**, and the "191M" on the
cards is that file count minus most of the duplicated bank.

Two other structural facts that matter for quantization:

- **Q/K/V are packed** into `mha.in_proj_weight (3072, 1024)`; there is no
  separate `q_proj`/`k_proj`/`v_proj` module to quantize one at a time.
- The per-layer residual scale `alpha ≈ 0.427` is **folded into** `out_proj` and
  `linear2` at load time, so those weight matrices are already scaled when the
  quantizer sees them.

## 0.3 The 1.0 bpw budget (`results/bt4/budget.json`)

1 sign bit per weight + one fp16 row scale per 2-D weight; everything not
quantized stays fp16. Computed on the loaded module tree, deduplicated.

| scope | quantized | at 1.0 bpw | left fp16 | artifact | vs fp16 |
|---|---|---|---|---|---|
| named only (attn q/k/v/out + MLP fc1/fc2) | 110.10M | 13.96 MB | 150.3 MB | 164.3 MB | 2.26× |
| **all block 2-D — "Scope 2", the Step 1 choice** | **149.91M** | **19.19 MB** | 70.7 MB | **89.9 MB** | **4.12×** |
| all 2-D (adds input_block + heads) | 184.67M | 23.63 MB | 1.2 MB | **24.8 MB** | 14.94× |

Scope 2 covers 81 % of the weights the model uses; the ~24 MB end state needs the
third row, of which `input_block.preprocess` (25.17M, 100 MB fp16) is the single
biggest item. BT4 has no vocabulary table, but it does have a board-wide dense
projection, so "no embedding surprise" would have been false in the way that
matters.

## 0.4 Position sets (`data/bt4/puzzles.jsonl`)

From the Lichess puzzle dump (6,193,530 rows; 1,419,269 rejected as duplicate
FEN, out of rating band, or with an illegal blunder/key move; 4,774,261 usable),
sampled deterministically by puzzle-id hash:

| split | n | rating p10/p50/p90 | solver is black |
|---|---|---|---|
| calib | 512 | 1156 / 1617 / 2321 | 241 (47.1 %) |
| test | 2000 | 1115 / 1589 / 2305 | 973 (48.7 %) |

Themes overlap heavily: `short` 1290, `middlegame` 1199, `endgame` 1194,
`crushing` 1085, `advantage` 857, `long` 770, `mate` 546, `master` 344.

Each row keeps the raw FEN, `first` (= `Moves[0]`, the opponent's blunder) and the
full `moves` list. The harness pushes `first` and grades on `moves[1]`, which also
gives `_leela_planes` one ply of real history instead of a cold FEN.

**Limitation, stated up front:** Lichess puzzles are tactical by construction.
There is no quiet/positional stratum here, so "spanning tactical and positional
play" is not satisfied by this set; positional coverage has to come from game
corpora (or from the Step 3 match positions).

## 0.5 FP16 baseline (`results/bt4/fp16_base_{test,calib}.json`)

1-node, searchless, legal-move-masked, on the position the solver actually faces:

| metric | test (n=2000) | calib (n=512) |
|---|---|---|
| **puzzle solve@1** (top-1 == `Moves[1]`) | **94.40 %** | 94.14 % |
| **puzzle solve@5** | **99.80 %** | 99.80 % |
| legal-move policy entropy | 0.796 nats (0.243 norm.) | 0.818 nats (0.249) |
| mean top-1 prob | 0.816 | 0.809 |
| legal moves per position | 30.70 (2–59) | 30.7 |
| WDL argmax W/D/L | 1418 / 415 / 167 | — |
| mean WDL probs W/D/L | 0.676 / 0.223 / 0.101 | 0.653 / 0.234 / 0.113 |
| mean q = P(win) − P(loss) | +0.575 | +0.540 |
| throughput (batch 64) | 1191.1 pos/s | 1002.8 pos/s |

This is the benchmark the plan wanted: a 1-node, searchless policy that solves
94 % of 1600-rated Lichess puzzles first try. Colour symmetry confirms the
mirroring is right (white and black solve at the same rate; mean q +0.575/-0.017
for solved/unsolved, i.e. the value head calls the solved positions winning).

The old 6.30 % was the opponent's move being graded: `Moves[0]` is a plausible
move by the side to move, so the prior ranked it 2nd–5th (60.65 % "solve@5") and
almost never 1st. That the near-miss rank histogram moved from
`[126, 359, 323, 232, 173, …]` to a 94 % hit rate is the strongest evidence the
convention is now right.

### Throughput (the reason this pivot is cheap)

| dtype | batch 32 | 64 | 128 | 256 | peak VRAM |
|---|---|---|---|---|---|
| fp16 | 1148.6 pos/s | 1286.2 | 1243.1 | 1239.2 | 0.68 GiB |
| fp32 | 402.2 pos/s | 396.6 | 399.9 | 402.8 | 1.34 GiB |

~1,250 pos/s fp16 at batch 256 means a 300k-position distillation pass is about
**4 minutes** of forward compute, and a 2,000-position evaluation is 1.7 s —
versus ~7 positions/s for the Qwen3-Reranker at 384 tokens. The plan's premise
that this runs orders of magnitude faster holds, measured.

## 0.6 Step 1 gates — frozen 2026-11-21, amended by A1 before any arm was scored

Arms on the identical 512 calibration positions, read out on the 2,000 test
positions, at **Scope 2** (149.91M) and 1.0 bpw: `itq`, `qubo_act`, `qubo_seq`,
plus `bit1` (added, see below).

- **G1 drift (primary):** relative output error of the quantised stream vs the
  FP16 stream after every block, on 128 test positions. `qubo_seq` must beat
  `qubo_act` on the last block, where drift compounds.
- **G2 policy agreement:** quantised legal-masked top-1 == FP16 top-1.
- **G3 legal-move KL:** mean KL(FP16 ‖ arm) over legal moves.
- **G4 usability (amended):** `solve@1 ≥ 0.66` (70 % of the 0.944 FP16 figure)
  **and** `solve@5 ≥ 0.90` (90 % of 0.998). The original `solve@5 ≥ 0.45` was
  derived from the pre-A1 0.6065 and is void; it was never used to judge an arm.
- **No abort rule** (per the request): a sharp zero-shot drop does not stop the
  arms; they proceed to Step 2. Only a harness fault — NaN readout, or G2 < 5 %
  agreement, which would mean the wiring is broken — aborts.

**`bit1` is an addition, not a substitution.** 1.0 bpw admits two very different
representations: a LittleBit low-rank path of rank `r = d_in·d_out/(d_in+d_out)`
(one sign matrix folded through two ranks, as in LittleBit-2) or a full-rank
elementwise `h_i · sign(W_ij) · g_j`. The first carries an error floor of
`sqrt(1 − r/d)` that exists because of the rank bottleneck, not because of 1 bit
of precision. `bit1` measures that floor so a collapse can be attributed
correctly. Both are exactly 1.0 bpw of sign bits; measured totals are 1.0415 bpw
(itq/qubo, including scales) and 1.0369 bpw (bit1).

**Known gap in this data:** the Lichess puzzle file carries no game outcome, so
"WDL accuracy" has no ground truth here. Step 0/1 report the WDL readout (argmax
distribution, mean q) and WDL *agreement* with FP16, not accuracy. True WDL
accuracy needs outcome labels or Stockfish labels, which is Step 3's setup.

## 0.7 Files

- `harness/bt4/audit.py` → `results/bt4/audit.json`, `results/bt4/_header.json`
- `harness/bt4/budget.py` → `results/bt4/budget.json`
- `harness/bt4/puzzles.py` → `data/bt4/puzzles.jsonl`
- `harness/bt4/base.py` → `results/bt4/fp16_base_{test,calib}.json`,
  `results/bt4/_cache/policy_fp16_{test,calib}.npy` (+ `value_`, `gold_idx_`)
- `harness/bt4/q1bit.py` → `results/bt4/step1_quant_*.json`, `step1_eval_*.json`
- checkpoint `data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz` (336 MB) and puzzle
  dump `data/bt4/lichess_db_puzzle.csv.zst` (310 MB); both are untracked, and
  nothing in this step is committed

The FP16 policy logits are cached per split so every quantized arm is scored
against exactly the same reference, not a recomputation.
