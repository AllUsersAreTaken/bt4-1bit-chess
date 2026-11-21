# BT4 at 1.04 bpw — Step 4: fix the training distribution, re-measure the rating

Step 3's finding was that the 1.0415 bpw student played `1. a3` and rated ~1025
Elo while its fp16 teacher played `1. d4` and rated ~2650. The hypothesis tested
here: that gap was **distribution shift**, not the bit — the student trained on
60,000 Lichess *puzzle* positions only, so openings, quiet play and endgames were
off-manifold, and 128k gradient steps rewrote the body to fit that manifold.

The hypothesis was half right, and the size of the effect is the interesting part.

## 4.0 Pool: 80 % real games, 20 % puzzles

`harness/bt4/games.py`, source `database.lichess.org` standard rated, 2013-01
(5,629 PGN games read, elos ≥ 1500, 24–160 plies).

| | |
|---|---|
| game positions | 102,400 |
| puzzle positions | 25,600 (re-seeded draw from the Step 2 pool) |
| total | **128,000**, `data/bt4/_planes_mixed.npy` (1.71 GiB) |
| ply p10 / p50 / p90 / p99 | **1 / 28 / 78 / 124** — move 1 through endgame |
| held-out eval | 2,437 positions from 60 *unseen* games |

Positions are built by **replaying the real game**, not from a bare FEN:
`_leela_planes()` fills its 8 history slots from `board.move_stack`, so a
FEN-rebuilt board would silently lose history and recreate the same class of bug
somewhere new. `data/bt4/step4_pool.log`.

New readouts added to `qat.py` and shown in every EVAL line:

* `game_agree` — top-1 agreement with the teacher on the 2,437 held-out game
  positions (the full-game number `agree` never measured);
* `game_kl`, `game_gold` — same positions, KL and top-1 vs the move actually played;
* `start` — the model's masked best move **on move 1**, the exact Step 3 failure.

## 4.1 The run

```
python harness/bt4/qat.py --pool mixed --arm qubo_act --steps 4000 \
    --batch 64 --every 200 --seed 0 --tag mix4000 --save
```

4,000 steps, batch 64 = 256,000 positions ≈ 2 epochs, **4330.4 s**, peak
**7.76 GiB**, no spill, `MIX_EXIT=0`. `results/bt4/step2_qat_mix4000.json`,
checkpoint `step2_weights_mix4000.pt`.

| step | solve@1 | agree | KL | **game_agree** | game_kl | game_gold | start |
|---|---|---|---|---|---|---|---|
| 0 | 0.2350 | 0.2025 | 2.2931 | 0.1871 | 1.4876 | 0.2060 | `h2h3` |
| 200 | 0.3925 | 0.3145 | 1.4693 | 0.2651 | 1.1695 | 0.2593 | `d2d4` |
| 400 | 0.5520 | 0.4380 | 0.9944 | 0.4165 | 0.8555 | 0.3578 | `f2f3` |
| 800 | 0.6365 | 0.5065 | 0.8246 | 0.4510 | 0.7347 | 0.3837 | `d2d4` |
| 1200 | 0.6945 | 0.5480 | 0.7538 | 0.4748 | 0.6922 | 0.3947 | `b1c3` |
| 1600 | 0.6980 | 0.5700 | 0.7189 | 0.4817 | 0.6638 | 0.4021 | `b1c3` |
| 2000 | 0.6920 | 0.5670 | 0.7029 | 0.4875 | 0.6849 | 0.3927 | `g1f3` |
| 2400 | 0.7110 | 0.5790 | 0.7237 | 0.4785 | 0.6792 | 0.3890 | `g1f3` |
| 2800 | 0.7115 | 0.5780 | 0.6970 | 0.4842 | 0.6823 | 0.3730 | `d2d4` |
| 3200 | 0.7260 | 0.5860 | 0.6435 | 0.4932 | 0.6347 | 0.4034 | `d2d4` |
| 3600 | 0.7200 | 0.5790 | 0.6540 | **0.5096** | **0.6204** | **0.4116** | `d2d4` |
| 4000 | 0.6855 | 0.5625 | 0.9487 | 0.4674 | 0.7616 | 0.3648 | `c2c3` |

**Move 1 is fixed, immediately.** Zero-shot the body says `h2h3`; by step 200 it
says `d2d4`, and from step 1200 on it only ever says `d2d4`, `g1f3` or `b1c3`.
The single clearest symptom of Step 3 disappears in 200 steps of real-game data.

**Full-game agreement plateaus at ~0.50, not 0.75–0.80.** It climbs fast to 0.451
by step 800, then grinds: 0.475 (1200) → 0.488 (2000) → 0.503 (2200) → 0.510
(3600). Two epochs over 102k real-game positions buys +32 pp over zero-shot and
then stalls. That plateau — not the missing distribution — is what cap the
rating gain below.

Note the trade: on the *puzzle* gate the mixed pool is at 0.6855–0.7305 versus
0.7600 for puzzle-only, i.e. −3 to −7 pp, exactly what moving 80 % of the pool
off puzzles should cost. At step 1400–1600 it is actually *ahead* of the
puzzle-only run (0.7035 vs 0.6575), so the game data is not hurting early
learning, only the final puzzle specialisation.

## 4.2 The rating

Same harness, same 1320 anchor, same openings/clock/seed as Step 3's 1025.

| player | sims | games | score | rating | 95 % CI |
|---|---|---|---|---|---|
| puzzle-only student (Step 3) | 1 | 32 | 0.156 (+2 =6 −24) | 1025 | [850, 1145] |
| **mixed student, greedy** | 1 | 32 | 0.391 (+8 =9 −15) | **1245** | [1130, 1340] |
| **mixed student, 64-node PUCT** | 64 | 16 | 0.750 (+11 =2 −3) | **1510** | [1340, 1790] |
| fp16 teacher, greedy | 1 | 16 | 1.000 at 1320+1600 | **2650** | [2495, 2950] |

`results/bt4/step3_elo_mix4000_1320.json`, `…_1320_puct64.json`,
`results/bt4/step4_elo_*.log`.

Decomposition of Step 3's ~1625-Elo gap:

| change | Elo | source |
|---|---|---|
| 1025 → 1245 | **+220** | training distribution (80 % real games) |
| 1245 → 1510 | **+265** | search (1 → 64 PUCT nodes, ~1.7 s/move vs 0.027 s) |
| 1510 → 2650 | **~+1140** | still unexplained: the bit, the budget, or the body |

So distribution shift was **real, measurable, and about 13 % of the gap.** The
move-1 catastrophe was a genuine off-manifold failure and fixing it is worth
~220 Elo — but it was a symptom of a wider weakness, not the whole weakness.

Search is worth more than the distribution fix (+265 for 64× the compute per
move), which fits Step 2's numbers: the student's *ranking* (`agree` 0.56–0.59)
is far better than its *first choice*, and search is exactly the tool that
converts ranking into choices. This is the same reason `solve@5 = 0.9735` looked
so good while `solve@1` was the truth.

## 4.3 What is honestly still unknown

1. **The checkpoint we played is the worst readout after step 3200.** Step 4000
   regressed on every metric versus step 3600 (game_agree 0.4674 vs 0.5096, KL
   0.9487 vs 0.6204, `start` back to `c2c3`). Intermediate checkpoints were not
   saved, so both Elo numbers were measured on the worse state — they are
   conservative, and "train longer" is not established as a win from this run.
2. **The PUCT number is 16 games, wide CI [1340, 1790].** Directionally solid
   (0.750 vs 0.391 at the same anchor), but +265 Elo is not tightly measured.
3. **The remaining ~1140 Elo is still unattributed.** This run did not add the
   control that could split it: the **same mixed pool, same loss, same 4,000
   steps, on the fp16 model**. If fp16-mixed lands near 2400, the bit costs
   ~900 in play; if it lands near 1500, then 4,000 steps of 1.04 bpw distillation
   is simply underfitting and the bit is nearly free. That is the next run, and
   it is the one this experiment needs — the user was right that an fp16
   *puzzle-only* control would have told us nothing, but an fp16 *mixed* control
   is now the decisive measurement.
4. `game_agree` ~0.50 at 1.04 bpw is the honest headline metric for play: half
   the teacher's first choices, over positions spanning move 1 to move 124.

## 4.4 The step-3600 → 4000 cliff, explained (and fixed)

Step 4000 was worse than step 3600 on every metric (game_kl 0.6204 → 0.7616,
game_agree 0.5096 → 0.4674, `start` `d2d4` → `c2c3`). Cause, from the code and
the training log:

* **There was no LR schedule.** `qat.py` ran a constant 1e-4 signs / 2e-4
  scales for all 4,000 steps — zero matches for any scheduler. Under a plain
  STE, constant LR means sign flips keep firing at the same rate right up to
  the last step, so the last 200 steps have exactly as much authority to leave
  a good basin as step 200 did.
* **The log shows the basin exit.** Last-block hidden loss (the relative MSE on
  the residual stream the heads read): 1.05 → 1.37 → 1.70 → 2.12 → 2.28 →
  **4.86** over the final 175 steps, while the policy loss stayed flat at
  ~0.6. Gradient norm peaked at **32.9** (step 3675) — clipping at 1.0 caps the
  step, so that is a symptom of a bad batch, not the cause.
* **There was nothing to fall back on.** Best-checkpoint saving did not exist,
  so the only artifact on disk was the final, regressed state.

The mechanism is worth keeping: the body can keep fitting the *training batches*
policy-wise while its residual stream walks away from the teacher, and the eval
(a held-out measure) is what sees it. Training loss was not the alarm; held-out
game KL was.

Fixed in `qat.py`:

| change | flag |
|---|---|
| cosine LR decay to a floor | `--lr_min` (default `1e-5`) |
| keep the best epoch by held-out game KL | `--save-best` |
| LR now printed in the step line | — |

Verified on a 200-step smoke run (`results/bt4/step4_smoke_sched.log`): LR fell
9.66e-5 → 8.68e-5, and `BEST` fired at step 200 writing
`step2_weights_smoke_sched_best.pt`; `best` is recorded in the run JSON.
The 4,000-step run was not re-run — the step-3600 state was never saved, so its
Elo numbers stay as measured on the worse checkpoint (conservative).

## 4.5 Is the WDL head why search underperforms? No.

`harness/bt4/_diag_wdl.py` on the 2,437 held-out game positions
(`results/bt4/step4_wdl_step2_weights_mix4000.json`):

| metric | value |
|---|---|
| WDL argmax agreement with teacher | **0.8096** |
| score correlation (P(win) − P(loss)) | **0.9203** |
| P(win) MAE | 0.1095 |
| value MSE | 0.0332 |
| draw rate, student vs teacher | 0.2925 vs 0.2931 |
| mean score, student vs teacher | 0.1832 vs 0.1107 (+0.07 optimistic) |

The value head is the **best-preserved** part of the model — 0.81 argmax
agreement and r = 0.92, against 0.47 policy `game_agree`. So the +265 Elo from
64 PUCT nodes was not limited by a noisy WDL head, and upweighting the value
loss is not where the remaining ~1140 Elo sits. The one bias worth noting is a
consistent **+0.07 optimism** in the expected score, which would make search
slightly over-confident in balanced positions.

What this leaves as the live question is **rank vs binarisation**, which §4.6
answers.

## 4.6 Rank vs binarisation: the control (and it does not say what we hoped)

`harness/bt4/qat.py --no-sign` runs the same factored `U Vᵀ` with the same rank
`d/2`, the same scales, the same pool and the same loss, but the factors stay
**continuous** — storage 2.40 Gbit, **16.04 bpw**, exactly the same 150.3M
parameters. It is a binarisation control, not a rank control: both arms are
rank-limited.

One trap, caught before the run meant anything: dropping `sign()` on the arm's
stored `U, V` starts from a **different weight matrix**, because those tensors
are the *pre-sign* latents. Zero-shot came out at 0.047 `solve@1` / 0.0455
`game_agree` / `g1h3` instead of 0.235 / 0.1871 / `h2h3`. Fixed by starting the
continuous path from the binary weight's own ±1 factors, after which zero-shot
reproduces the binary arm **exactly** (0.2350 / 0.2025 / 2.2931 / 0.1871 /
1.4876 / 0.2060 / `h2h3`). Same W, same rank, only the constraint removed.

1,000 steps, cosine to 1e-5, best-checkpoint on (`results/bt4/step4_rank1000.log`,
`step2_qat_rank1000.json`, 519.7 s, 7.19 GiB):

| step | binary `game_agree` | **fp low-rank** | binary `solve@1` | **fp low-rank** |
|---|---|---|---|---|
| 600 | 0.4391 | 0.3947 | 0.6085 | 0.5255 |
| 800 | 0.4510 | 0.4062 | 0.6365 | 0.5540 |
| 1000 | 0.4579 | **0.4169** | 0.6605 | 0.5550 |
| plateau | ~0.51 (4k steps) | **~0.42** (1k steps) | 0.73 | 0.56 |

| player | rating | 95 % CI |
|---|---|---|
| fp low-rank, greedy, 1,000 steps | **1145** | [1025, 1245] |
| 1-bit, greedy, 4,000 steps (§4.2) | 1245 | [1130, 1340] |
| 1-bit + 64 PUCT (§4.2) | 1510 | [1340, 1790] |

**Removing the sign constraint made the model worse, not better** — on every
metric, at every matched step, and on the rating (1145 vs 1245, with the control
additionally favoured by having 4× less training but a tighter CI). So the
user's first branch does not hold: rank `d/2` did **not** have capacity waiting
to be unlocked by precision. The honest reading is the second branch — **the
~0.50 `game_agree` ceiling is shared by a full-precision model, so the bottleneck
is not the one bit.** It is the rank `d/2` bottleneck and/or the distillation
budget, loss balance and pool, all of which this control held fixed.

Caveats, stated plainly:

* The control stopped at 1,000 steps (519.7 s). It has 4× less budget than the
  binary run, so "plateaus lower" and "is undertrained" are not separated here;
  what *is* separated is that full precision buys nothing at matched steps.
* Schedules differ: the binary 4,000-step run used a constant LR (the bug in
  §4.4), the control used cosine to 1e-5. By step 700 the control's LR was
  2.85e-5, so part of its late flatness is schedule, not capacity.
* One practical surprise worth recording: the **binary path is ~2× slower per
  step** than the continuous one (0.78–1.09 s vs 0.41–0.51 s at batch 64). The
  `sign` is extra elementwise work over 150M latents, so at 1.04 bpw you pay in
  step time what you save in bytes.

Next lever, therefore, is not more bits and not more precision: it is **rank**
and **budget/loss balance**.

## 4.7 How much rank `d/2` actually costs (exact SVD, zero training)

`harness/bt4/_diag_svd.py` replaces every Scope-2 matrix with its exact rank-r
SVD truncation in fp16 — the *best possible* rank-r approximation of the
teacher's weights — and evaluates it. No training, no signs (`--mult` scales the
rank; `1x` is the 1.04 bpw path's own rank).

| rank | `--mult` | factor bpw | mean rel. Frobenius err | solve@1 | solve@5 | game_agree | move 1 |
|---|---|---|---|---|---|---|---|
| `d/4` | 0.5 | 8.0 | 0.5347 | 0.1175 | 0.4015 | 0.1112 | `c2c3` |
| **`d/2` (the 1.04 bpw rank)** | 1 | 16.0 | **0.1953** | **0.3110** | 0.7945 | **0.2179** | `b1c3` |
| full (`d`) | 2 | — | 0.0002 | **0.9440** | **0.9980** | **0.9992** | `d2d4` |
| full (`d`, saturated) | 4 | — | 0.0002 | 0.9440 | 0.9980 | 0.9992 | `d2d4` |

Two things this settles:

1. **The diagnostic is sound.** At full rank the truncation reproduces the fp16
teacher exactly — solve@1 0.9440, solve@5 0.9980, `game_agree` 0.9992, move 1
`d2d4`. The 0.0002 residual error is fp16 rounding. So the numbers at lower rank
are the real effect of removing rank, not a harness artifact.

2. **BT4's Scope-2 weights are nowhere near rank `d/2`.** Exact SVD at the
1.04 bpw rank throws away **19.5 %** of each matrix (Frobenius, mean; up to
31.9 %) and drops the model from 0.9440 to **0.3110** solve@1 and from 1.0 to
**0.2179** `game_agree` — *at 16 bits per factor*. Halving that rank again
(`d/4`) takes it to 0.1175 / 0.1112.

Why this matters for the bit-vs-rank argument: **the same 1.04 bpw buys either a
rank-`d/2` factorisation or 1 bit per weight at full rank, and the rank choice
throws away far more than the bit does.** The trained 1-bit low-rank model
reaches 0.6855 solve@1 / 0.4674 `game_agree`, i.e. QAT recovers a great deal of
what truncation destroyed — but it is recovering *inside* a subspace the teacher
does not live in.

Honest limit on that reading: exact SVD is the best *weight-space* approximation,
not a hard behavioural ceiling. A trained rank-r model redistributes capacity and
can beat it, which is exactly what training does here (0.3110 → 0.6855). So this
does not prove the rank-r family caps at 0.47; it proves the rank-r *copy of the
teacher* is badly damaged, and that rank — not the sign — is where the structural
loss starts.

The falsifiable next step this points to: **`bit1`, full rank, 1.0 bpw, no rank
truncation, trained on the same mixed pool with the fixed cosine schedule.**
Prediction if rank is the wall: full-rank 1-bit clears 0.60 `game_agree` at
1.0 bpw and beats the 1.04 bpw low-rank model on Elo despite a *lower* zero-shot
(0.0950 vs 0.2350). If it instead stalls near 0.47 as well, the wall is the
distillation budget/loss and neither rank nor bits.
