# BT4 — Step 5: full-rank 1-bit (`bit1`) vs the rank-`d/2` path

Prediction under test (stated before the run): *if rank `d/2` is the wall, full-rank
1-bit at 1.0 bpw clears 0.60 `game_agree` and beats the 1.04 bpw low-rank model on
Elo, despite a lower zero-shot.*

**Result: it does not. The prediction is falsified.**

Run: `--arm bit1 --pool mixed --steps 4000 --batch 64 --seed 0 --lr_min 1e-5
--save-best`, **4045.6 s**, peak **7.75 GiB**, `bit1` at **1.0369 bpw** (one sign
per weight + fp16 row/col scales, **no rank truncation**).
`results/bt4/step2_qat_bit1_4k.json`, `step5_qat_bit1.log`.

## 5.1 Head to head, same pool, same budget, same seed

| model | bpw | rank | zero-shot solve@1 | 4k solve@1 | 4k `game_agree` | KL | **Elo** |
|---|---|---|---|---|---|---|---|
| `qubo_act` factored | 1.0415 | `d/2` | 0.2350 | **0.6855** | **0.4674** | 0.9487 | **1245** [1130, 1340] |
| `bit1` full rank | **1.0369** | full | 0.0780 | 0.4580 | 0.3529 | 1.2398 | **980** [795, 1100] |

bit1 is worse on **every** metric: −22.7 pp `solve@1`, −11.5 pp `game_agree`,
−0.29 nats KL, and **−265 Elo**. It even scores below the puzzle-only student of
Step 3 (1025), despite having both the better pool *and* the better rank budget.

`bit1` trajectory (`game_agree`): 0.078 (0) → 0.144 (200) → 0.126 (400) →
0.189 (600) → 0.264 (800) → 0.291 (1000) → 0.318 (1200) → 0.336 (2000) →
**0.353 (4000)**.

## 5.2 Why full rank lost

The 4,000 steps are not the story; the **initialisation** is.

* `sign(W)` at full rank leaves **60.9 %** Frobenius error on `blk00.ff2` (61 % is
  typical across the matrices) — it throws away every weight's magnitude and
  keeps only a rank-1 row/column scale.
* The littlebit path starts near **14 %** error, and its factors come from ITQ /
  QUBO refinement against the *activation covariance*, not against W elementwise.
* Consequence in the loss: bit1's policy KL sat at **2.47 for the first ~500
  steps** (step 0: 2.4728, step 400: 2.4669 — 0.006 nats in 400 steps), while the
  factored arm had already gone 2.2931 → 0.9452 by step 400. bit1 only began to
  move after ~step 500, and then crawled.

So a 19.5 % *static* rank tax is more than repaid by a well-conditioned starting
point. Rank is a real cost (Step 4 §4.7) but it is not the binding constraint:
removing it entirely, while also accepting a 61 %-error initialiser, is a large
net loss.

One bug fixed on the way, worth recording: the 1.04-bpw arms' scales live below
1, so the fixed `--scale_max 4.0` clamp was harmless there — but bit1's ALS scales
**start at 4.083**, so the clamp pinned them on step 1 and their gradient never
landed (first bit1 run: KL flat at 2.46). The clamp is now
`max(scale_max, 4 × init_max)`, recorded in `head.scale_init_max` /
`head.scale_clamp`. The rerun with the fix produced a nearly identical first 400
steps, so the clamp was not the main cause — but it was a real defect.

## 5.3 What this leaves

Three hypotheses have now been tested and two are dead:

| hypothesis | test | verdict |
|---|---|---|
| 1 bit is the wall | 16-bit low-rank control (§4.6) | **dead** — 16 bpw did not help |
| rank `d/2` is the wall | full-rank 1-bit (§5.1) | **dead** — removing rank hurt |
| the initialiser is the wall | bit1's `sign(W)` start vs ITQ/QUBO | **live** |

The live hypothesis is specific and cheap to test: bit1 failed because its signs
are aligned to **W elementwise**, ignoring the activation covariance. The fix is
the row-wise QUBO initialiser — for row `i` of the full-rank binary matrix,

    min_{b_i ∈ {±1}^d_in}  (w_i − h_i b_i ⊙ g)ᵀ Σ (w_i − h_i b_i ⊙ g)

which decouples per row into a native QUBO with `Q = diag(g) Σ diag(g)`. Same
1.04 bpw, same full rank, **no SVD and no rank truncation**, but signs chosen
against the activations the layer actually sees. That is `bit1_qubo`, and it is
the arm the evidence now points at.

If `bit1_qubo` also stalls near 0.35, then neither bits, rank nor initialiser is
the wall and the remaining ~0.47 `game_agree` ceiling is the distillation budget
and loss balance — at which point the right spend is game-position distillation
with more steps and a tuned loss, not another quantisation scheme.

## 5.4 Amendment (2026-11-21): `bit1` never moved a single sign

Run before spending any QAT time on `bit1_qubo`, as a gate on the initialiser
hypothesis. Both results change how §5.2–5.3 must be read.

### 5.4.1 Sign flips: exact zero

`harness/bt4/_diag_flip.py` → `results/bt4/step5_flip_diag.json`. `quantise_bit1`
keeps `s = sign(W)` and fits only the scales, so bit1's init signs are
deterministic and the checkpoint can be compared against them exactly:

| quantity | value |
|---|---|
| signs in Scope 2 | 149,913,600 |
| signs the 4,000-step run moved | **0** (every one of the 105 matrices: 0) |
| mean relative change in `h` | 0.2145 |
| mean relative change in `g` | 0.0208 |
| mean relative change in the effective weight `h s g` | 0.2172 |

The "full-rank QAT" run was therefore a **frozen 1-bit sign pattern plus its
diagonal scales**: it never searched the sign space, so its 0.3529 `game_agree`
is the ceiling of `sign(W)` with fitted row/column scales, not the ceiling of
full-rank 1-bit QAT.

Mechanism: `BitFull` stored `part.S`, the ±1 pattern itself, so every latent began
at |B| = 1.0 exactly. The run's Adam travel is Σlr ≈ **0.22** (cosine, 1e-4 → 1e-5,
4,000 steps), i.e. a latent can cover ~22 % of the distance needed to cross zero.
Share of latents inside that budget:

| latent initialisation | |latent| median | inside 0.22 budget |
|---|---|---|
| bit1 `part.S` | 1.0 | **0.0 %** |
| factored `_Path` `\|U\|` | 0.0395 – 0.0686 | **99.5 – 100 %** |
| factored `_Path` `\|V\|` | 0.0327 – 0.0881 | **98.4 – 100 %** |
| bit1 pre-sign `W/(h g)` | — | **13.6 %** (mean over matrices) |

Cross-check that factored signs really do move: two runs from the identical
`qubo_act` init (`full_qubo_act_ckpt`, `mix4000`) end on **2,532,643 / 149,867,040
= 1.69 %** different signs. That is an upper bound on either run's flips, not a
measurement of one, but it is nonzero where bit1's is exactly zero.

**Correction to §5.3.** The row "rank `d/2` is the wall — dead" is not supported
by the bit1 run. That run tested *`sign(W)` + diagonal scales*, so full rank at
1 bpw is still **untested** while the signs are frozen. What §5.1 does establish
is narrower and still holds: a full-rank 1-bit matrix started at `sign(W)` is a far
worse starting point than the ITQ/QUBO path (0.0780 vs 0.2350 zero-shot; 61 % vs
~14 % weight error), and diagonal scales alone cannot close that gap.

### 5.4.2 `bit1_qubo` zero-shot gate

`harness/bt4/bit1_qubo.py` → `results/bt4/step5_bit1_qubo_gate.json`, **113.6 s**,
init saved to `results/bt4/step5_bit1_qubo_init.pt` (430 MiB: signs, scales and
pre-sign latents). Per-row QUBO against the clean-activation Σ (512 calib
positions), 1 round, `d_in` 1-opt sweeps, 4 restarts, 1.0369 bpw.

| metric (weighted over 149.9 M weights) | `sign(W)` + ALS | `bit1_qubo` |
|---|---|---|
| activation error ‖(W−Ŵ)X‖/‖WX‖ | 0.5303 | **0.4041** (−23.8 % rel.) |
| Frobenius error ‖W−Ŵ‖/‖W‖ | 0.6024 | 0.9571 |
| signs changed by the QUBO | — | 29.6 % |

99 of 105 matrices improve in activation error. Σ is far from isotropic — mean
‖Σ − diag Σ‖/‖Σ‖ = **0.789** — so inter-channel cancellation is available, and it
does beat the 60.3 % per-entry magnitude floor *in the activation metric*. The
Frobenius column moving the other way is expected: this objective does not target it.

Zero-shot on the same 2,000-puzzle readout as every other arm:

| init | solve@1 | solve@5 | agree | KL | game_agree | game_kl | move 1 |
|---|---|---|---|---|---|---|---|
| `bit1` (`sign(W)`) | 0.0780 | 0.2875 | 0.0625 | 2.4728 | 0.0780 | 1.5867 | h2h3 |
| **`bit1_qubo`** | **0.1140** | **0.3770** | 0.0355 | 2.4737 | 0.0648 | 1.5874 | g2g3 |
| `qubo_act` factored | 0.2350 | 0.4790 | 0.2025 | 2.2931 | — | — | — |

Verdict against the gate as stated before the run ("proceed to 4,000-step QAT if
solve@1 reaches ~0.18"): **it does not.** solve@1 rises 4.6 pp (0.0780 → 0.1140, a
third of the distance to the factored arm's 0.2350, unpaired across 2,000 common
puzzles), while `agree` and `game_agree` are flat-to-worse and terminal policy KL is
unchanged (2.4737 vs 2.4728). Locally the initialiser does what it claims; end to
end it recovers no KL at all. The 70-minute QAT run is therefore **not** authorised
by this gate.

### 5.4.3 What the gate points at instead

Σ was captured on the **clean** fp16 stream and all 105 matrices were replaced at
once, so after the first substitution every downstream layer sees a different input
distribution and the per-matrix optimum stops being the end-to-end optimum. The
factored arm already has that control (`qubo_seq`: layer by layer, Σ taken from
the stream the quantised model actually emits); full rank never ran it.

That is the cheap next test: sequential Σ for `bit1_qubo`, one extra pass over 512
positions per matrix, a few minutes and no training. It separates "the initialiser
does not help" from "the initialiser was fitted to the wrong distribution".

If it recovers a useful part of the gap, the QAT run has a defensible start:
`BitFull(latent_init="ratio")` now stores `W/(h g)`, the saved init carries those
latents (median |latent| ≈ 0.4–0.6, 13.6 % inside the 0.22 budget, and 0
mismatches against the stored signs, so the checkpoint *is* the QUBO model at
evaluation), and a 4,000-step run can then actually flip signs early.

One accounting note for any comparison here: the 1.0415 bpw factored baseline in
§5.1 (0.4674 `game_agree`, 1245 Elo) is the regressed step-4000 checkpoint with no
LR decay. Its own best epoch was **0.5096 `game_agree` / 0.7200 solve@1 at step
3600**, so the fair yardsticks are that peak, or a cosine-LR rerun of `qubo_act`.

### 5.5 Sequential output-matching QUBO (2026-11-21)

The §5.4.2 gate fitted every matrix against the clean stream's Sigma and replaced
all 105 at once, which is the objective that sank `qubo_seq` in Step 1. This run
applies both corrections, still with no training: **sequential** (each matrix is
quantised in place before the next capture, so X~ is the stream the quantised
model actually emits) and **output-matching**

    min || Y_clean - W_hat X~ ||_F^2,   Y_clean = W X_clean

whose expansion keeps `Q = diag(g) Sigma~ diag(g)` and swaps the linear target to
the cross-covariance `C = Y_clean X~^T = W (X_clean^T X~)`.

`harness/bt4/bit1_qubo_seq.py` → `results/bt4/step5_bit1_qubo_seq.json`,
**288.5 s** (279.4 s capture), 512 calib positions, 1.0369 bpw, `--ridge 0`.

| aggregate (weighted over 149.9 M weights) | `sign(W)`+ALS | sequential out-match |
|---|---|---|
| output error ‖Y_clean − ŴX~‖/‖Y_clean‖ | 0.7367 | **0.4522** (105/105 improve) |
| activation error on the corrupted stream | 0.5321 | 0.4701 |
| activation error on the clean stream | 0.5322 | 0.4870 |
| Frobenius error | 0.6024 | 0.9353 |
| signs changed | — | 25.7 % |

The clean-stream activation error is *worse* than §5.4.2's 0.4041, and that is the
point: this objective spends its freedom matching the clean **output** on the
corrupted input rather than imitating the teacher on a stream the model no longer
produces. The metric that predicts zero-shot is the output one, not the clean one.

**Ridge answer.** A `lambda = 0.02` ridge on `Sigma~` changes essentially nothing:
0.452189 vs 0.452186 on the output objective, 0.935195 vs 0.935336 on Frobenius,
and 43/105 matrices "better" at the 6th decimal. Damping is a no-op at this
strength; the Frobenius blow-up is not an ill-conditioning artefact, it is simply
what this objective looks like when measured in the wrong norm.

Zero-shot, same 2,000-puzzle readout as every arm:

| init | solve@1 | solve@5 | agree | KL | game_agree | game_kl | move 1 |
|---|---|---|---|---|---|---|---|
| `bit1` (`sign(W)`) | 0.0780 | 0.2875 | 0.0625 | 2.4728 | 0.0780 | 1.5867 | h2h3 |
| `bit1_qubo` clean Sigma (§5.4.2) | 0.1140 | 0.3770 | 0.0355 | 2.4737 | 0.0648 | 1.5874 | g2g3 |
| **`bit1_qubo_seq`** | **0.1605** | 0.3785 | **0.1285** | **2.3322** | **0.1440** | 1.5739 | f2f3 |
| `qubo_act` factored (zero-shot) | 0.2350 | 0.4790 | 0.2025 | 2.2931 | — | — | — |

So the two corrections together recover **0.14 nats** of the KL the clean-Sigma
gate recovered none of, double `agree` (0.0625 -> 0.1285) and nearly double
`game_agree` (0.0780 -> 0.1440). The arm now sits 0.039 nats from the factored
arm's zero-shot KL, against 0.18 nats for the clean-Sigma version.

Against the gate as stated ("clears ~0.18 solve@1 **and** cuts KL meaningfully"):
KL is cut meaningfully, solve@1 misses by 2.0 pp. Not a pass, not a clean fail —
the initialiser hypothesis is alive and the wrong-distribution diagnosis was the
real defect.

**Flip budget, measured.** `step5_bit1_qubo_seq_init.pt` stores `W/(h g)`; the
share of those latents inside the 0.22 Adam budget is 34.7 % at scale 1, **69.0 %
at scale 1/5**, 92.8 % at 1/10 — so `BitFull(latent_scale=0.2)` puts a 4,000-step
run on roughly BitPath's footing (98–100 %) before any QAT is spent.

**Amended 2026-11-21 (see §5.6): that inference was wrong.** A small positive
`latent_scale` maximises the *count* of flip-reachable signs, not their safety: it
brings ordinary-magnitude weights within reach, so the flips it enables are
expensive. The first run at `latent_scale=0.2` flipped 10.4 % of all signs inside
200 steps and erased the init's entire KL gain.

**Caveat.** The fused MHA kernel never calls `out_proj` (verified: 0 pre-hook
calls), so both models were patched with a dense explicit-SDPA forward that does,
checked against the fused logits first: max |dlogit| / max |logit| = 2.84e-3. Both
streams use the same patched forward, and the evaluated student uses
`install_full`'s explicit SDPA as every other arm does, so the comparison holds;
the patch is not bit-identical to the fused path.

### 5.6 `bit1_qubo_seq` QAT at `latent_scale=0.2` (2026-11-21, 400/4,000 steps)

`--init step5_bit1_qubo_seq_init.pt --latent-scale 0.2 --pool mixed --steps 4000
--batch 64 --every 200 --lr_sign 1e-4 --lr_scale 2e-4 --lr_min 1e-5 --save-best`
→ `results/bt4/step5_qat_bit1_qubo_seq_4k.log`.

**Step 0 reproduces the gate exactly**, which validates the `--init` path:
solve@1 0.1595 (gate 0.1605 fp16), KL 2.3320 (2.3322), agree 0.1275 (0.1285),
`game_agree` 0.1440, move 1 `f2f3`.

Then, within 200 steps:

| step | flip_frac (vs init) | flip_churn (vs previous readout) | solve@1 | agree | KL | game_agree | move 1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.1595 | 0.1275 | 2.3320 | 0.1440 | f2f3 |
| 200 | **0.1036** | 0.1036 | 0.0785 | 0.0555 | **2.4737** | 0.0685 | b1a3 |
| 400 | 0.1036 | **0.0011** | 0.0595 | 0.0470 | 2.4735 | 0.0644 | c2c4 |

15,528,866 signs moved in the first 200 steps, and then flipping essentially
stopped. Training loss agrees: policy 1.59 (step 25) -> 1.94 (300) -> 1.81 (400),
hidden 0.95 -> 1.40 -> 1.23, i.e. worse than the init it started from.

**Mechanism.** Liable latents are `W/(h g)` scaled by 0.2, so `|B| < 0.1` — about
10 % of entries — is inside a ~200-step Adam walk. Those are not the negligible
tail at this scale: at `latent_scale=0.2`, `|B| < 0.22` covers `|W|/(h g) < 1.1`,
i.e. *ordinary* weights, and each flip changes `W_ij` by `2 h_i g_j`, a fixed per-entry jump that does **not** depend on `|W_ij|` (corrected 2026-11-21).
15.5 M simultaneous flips of ordinary weight mass is a large, non-local
 perturbation, and it lands the model on KL 2.4737 — bit1's frozen-sign regime
(2.4728) — having spent the 0.14 nats the QUBO init had bought.

So the flip-budget quantity that matters is **weight mass per flip**, not the
fraction of latents inside the travel budget. `BitPath`'s latents are not
comparable at all: one `|U|` entry of ~0.05 modulates `r` weights through `l·V`,
so its small latents are cheap *and* its flips are structured. Full rank has one
bit per weight and a fixed jump per entry, so the latent magnitude only decides
*which* entries are reachable, never how large the jump is.

The untested configuration that follows from that: **`latent_scale=1.0`** (the
stored latents as-is). There only `|latent| < 0.22` is reachable — 13.6 % of
signs, and those are the near-zero weights, where `2|W|` is negligible; the
predicted churn (~0.1 % per 200 steps seen after the burst) then matches
BitPath's 1.69 %-over-4,000-steps scale instead of exceeding it 60-fold.
Run at 400/4,000 was still in flight when this was written; its likely finish is
a bit1-like 0.35 `game_agree` (prediction, not a result) because the burst has
already put it in the frozen-sign regime.

### 5.7 `latent_scale=1.0` gate: killed at step 400 (2026-11-21)

Same init and schedule as §5.6, only the latent rescale changed (1.0 instead of
0.2), run on the real 4,000-step cosine schedule so step 400 sees the true LR.
Pre-agreed gate: keep it if KL breaks clearly below 2.332 with modest steady
churn; kill and pivot to (1) if KL returns to ~2.47.

| step | flip_frac | churn | solve@1 | agree | KL | game_agree | move 1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.1595 | 0.1275 | 2.3320 | 0.1440 | f2f3 |
| 200 | 0.0217 (3,251,416) | 0.0217 | 0.0785 | 0.0510 | 2.4739 | 0.0644 | g1h3 |
| 400 | 0.0218 | 0.0015 | 0.0750 | 0.0375 | 2.4736 | 0.0685 | g1f3 |

**Killed at step 400, pivoting to the cosine-LR factored baseline.** KL did not
break the init; it went back to the plateau.

What this establishes, at both latent scales:

* The burst has the same shape. Every reachable latent flips inside the first ~200
  steps, then churn falls ~14x (2.17 % -> 0.15 % per 200 steps). The rescale changes
  the *size* of the burst (10.36 % -> 2.17 %, i.e. the tail `|W|/(h g) < ~0.02`) but
  not the shape.
* Even 2.17 % of simultaneous flips erases the 0.14-nat QUBO gain and lands on
  KL 2.4737 +/- 0.0002: bit1's frozen-sign plateau to four decimals, from three
  independent runs (scale 0.2, scale 1.0, and bit1 with no flips at all).
* So the latent scale is not the lever for steady sign search. The per-200-step flip
  budget is set by the `|W|/(h g)` tail, and any tail inside Adam's reach flips at
  once; steady flipping would need a sign-specific optimiser schedule or threshold,
  not a rescale.

Honest caveat on the rule itself: KL 2.47 at step 400 is also where `bit1` sat until
roughly step 600 before descending to 1.2398 at step 4,000, so this number alone
does not prove the run was unsalvageable. It proves the *init* was gone by step 200,
which is what the gate was built to detect. Continuing would have measured
"flippable-sign QAT from a QUBO seed", not "QAT that keeps its seed" — and only the
latter decides full rank versus factored.

### 5.8 Cosine-LR `qubo_act` yardstick, full 4,000 steps (2026-11-21)

The matched-schedule reference the bit1 gates are scored against: the `qubo_act`
factored arm (bpw 1.0415, `scale_init_max` 0.776) with its own LR made cosine
(1e-4 -> 1e-5) instead of constant, everything else identical. Step 0 reproduced
the original arm exactly (0.2350 / 0.2025 / KL 2.2931 / 0.1871), so the two curves
are comparable eval-for-eval. 4,000 steps, `WALL` 4,387.8 s (73.1 min), peak
7.77 GiB, 21 evals, `nonfinite` 0 throughout, `scale_max` 0.766 to the end.

| step | solve@1 | agree | KL | game_agree | game_kl |
|---|---|---|---|---|---|
| 0 | 0.2350 | 0.2025 | 2.2931 | 0.1871 | 1.4876 |
| 1000 | 0.6405 | 0.5205 | 0.8250 | 0.4567 | 0.7093 |
| 2000 | 0.7000 | 0.5770 | 0.6831 | 0.4908 | 0.6344 |
| 3000 | 0.7290 | 0.5960 | 0.6315 | 0.5146 | 0.6062 |
| 3600 | 0.7340 | 0.5980 | 0.6072 | 0.5146 | 0.5860 |
| 3800 | **0.7410** | 0.6020 | 0.6044 | 0.5137 | 0.5869 |
| 4000 | 0.7355 | 0.5970 | **0.5993** | **0.5158** | **0.5813** |

Result: `game_agree` 0.5158 and KL 0.5993 at step 4,000, i.e. the final eval is
also the `game_agree` peak, and it ends on the lowest `game_kl` of the run
(0.5813) — though not monotonically: from step 2,600 on the curve wobbles in a
flat band with four small upticks (largest 0.0036, at 3,200 -> 3,400), against a
0.865 drop over steps 0-2,400 and 0.022 over the last per-200-step stretch that
still fell with margin (2,400 -> 2,600). So the saved best checkpoint is step 4,000, but on
wobble-level evidence, not a clean descent. No gate applied to this run; it exists
to be the reference.

Step-matched against the constant-LR 4,000-step run (`step2_qat_mix4000.json`,
same init, same eval sets):

| step | const-LR solve@1 / gagree / KL | cosine solve@1 / gagree / KL | d KL (cos better) |
|---|---|---|---|
| 800 | 0.6365 / 0.4510 / 0.8246 | 0.6105 / 0.4419 / 0.8777 | -0.0531 |
| 1600 | 0.6980 / 0.4817 / 0.7189 | 0.6920 / 0.4785 / 0.7126 | +0.0063 |
| 2400 | 0.7110 / 0.4785 / 0.7237 | 0.7105 / 0.4891 / 0.6643 | +0.0594 |
| 3200 | 0.7260 / 0.4932 / 0.6435 | 0.7335 / 0.5076 / 0.6008 | +0.0427 |
| 3600 | 0.7200 / 0.5096 / 0.6540 | 0.7340 / 0.5146 / 0.6072 | +0.0468 |
| 4000 | 0.6855 / 0.4674 / 0.9487 | 0.7355 / 0.5158 / 0.5993 | +0.3494 |

What the yardstick establishes:

* Cosine costs 1.3-3.6 pp of `solve@1` over the first third (steps 400-1,400, worst
  -0.0355 at 600) and earns it back from step 1,800 on, after which it is ahead on
  every eval but one (step 2,400, behind by 0.0005). Constant LR gets its early
  progress from a schedule that is still at full LR when cosine has already decayed.
* It removes the tail collapse. From step 3,600 to 4,000 the constant-LR run lost
  0.0422 `game_agree`, 0.0345 `solve@1` and gained 0.2947 KL (its KL ends *above*
  where it started the tail, a divergence, not a plateau); cosine moved +0.0012 /
  +0.0015 / -0.0079 over the same window. This is the failure mode the earlier 4k
  number's "peak at step 3,600" was hiding.
* KL is the metric that separates them cleanly: cosine is ahead at every eval from
  step 1,600 to the end. The gap wobbles (0.02 to 0.11 across steps 1,800-3,800) but
  never closes, and it ends 0.35 wide because the constant-LR run's KL
  turns upward in its last 200 steps while cosine's keeps easing down.

Caveats. One seed and one arm, so this is a reference curve, not a claim that
cosine is better for every arm. On the `game_agree` column, single-eval differences
below ~0.02 are inside the binomial noise of a 2,437-game eval (+/-2 sigma ~ 0.020
at p ~ 0.51); the KL column and the 3,600->4,000 tail are outside it, and those are
what the conclusion rests on. And this run is the *factored* arm at 1.0415 bpw: its
0.5158 `game_agree` is the same-metric reference that `bit1`'s 0.3529 (`bit1`, 4k,
frozen signs) has to be read against, not a bit1 result.

### 5.9 Elo of the cosine yardstick (2026-11-21)

Played with the same protocol as every earlier 1.0415 bpw number (ladder 1320,
seed 7, `max-ply` 140, `sf-movetime` 0.02, UCI_LimitStrength anchors), so the
rows below are directly comparable. Checkpoint: `step2_weights_qubo_act_cos_4k_best.pt`
(= step 4,000, the last eval).

| 1.0415 bpw checkpoint | greedy 1-node | 64-node PUCT |
|---|---|---|
| `mix4000`, constant LR | 1245 [1130, 1340], 32 g, score 0.391 | 1510 [1340, 1790], 16 g, score 0.750 |
| `qubo_act`, cosine LR 4k | **1495 [1385, 1635]**, 32 g, score 0.734 | **1660 [1455, 3595]**, 16 g, score 0.875 |
| fp16 teacher (reference) | 2650 [2495, 2950] | — |

Greedy: +250 Elo, 21-5-6 vs SF-1320 (27 decisive). Same 32-game protocol and seed as
the 1245 run, and the two intervals are disjoint (the new lower bound, 1385, is 45
above the old upper bound, 1340; z ~ 3.0 on the difference), so the gain is outside
the estimator's noise. This is the number to trust.

PUCT-64: +150 Elo, 14-0-2 (all 16 decisive). Same 16-game protocol as the 1510 run.
Treat it as a point estimate only: at a single 1320 rung, a 0.875 score sits high on
the logistic, so the fit's upper bound (3595) is uninformative and the honest read is
"1660, lower bound 1455". One extra rung (1600) with 16 games would pin it.

Search is worth less to the better-calibrated net: greedy -> PUCT-64 adds +265 Elo to
`mix4000` (1245 -> 1510) but +165 to the cosine arm (1495 -> 1660), so under PUCT the
gap narrows from 250 to 150. The straightforward reading is that 64-node search
compensates for worse priors, and that part of the cosine gain is priors that no
longer need search to cash out. Either way the ordering survives the change in
protocol, which is what matters.

Where that leaves 1.0415 bpw on this card, greedy play: 1495 Elo, 1155 below the fp16
teacher and 250 above the same arm trained without LR decay. All numbers are this
RTX 4060 + Stockfish 19 at 20 ms; do not extrapolate.

## 5.10 Close-out: what BT4 established

Four results, each with its own evidence above. Nothing here is inferred from a run
that was killed or extrapolated off this machine.

1. **QUBO initialisation beats ITQ, and the lead survives QAT.** Zero-shot the
   `qubo_act` init is ahead of ITQ on every metric (solve@1 0.1915 vs 0.1375, agree
   0.169 vs 0.119, KL 2.3191 vs 2.4711; `step1_eval_qubo_act.json` vs
   `step1_eval_itq.json`), and on matched 2,000-step QAT it is still ahead at the
   end (solve@1 0.760 vs 0.673, KL 0.6251 vs 0.8865; `step2_qat_full_qubo_act.json`
   vs `step2_qat_full_itq.json`). QAT closes part of the gap, not all of it.
2. **Factored `BitPath` (U V^T) beats full-rank 1-bit under standard STE.** The
   factored latents move continuously and keep the QUBO basin; the full-rank arm has
   to cross a fixed `2 h_i g_j` jump per sign, and every burst that crossed one
   destroyed the seed (§5.4, §5.6, §5.7). `bit1` at 4k: `game_agree` 0.3529, 980 Elo
   [795, 1100], with 0 of 149,913,600 signs ever flipped (§5.4.1).
3. **Sequential output-matching fixes the clean-Sigma distribution mismatch**
   zero-shot. Clean-Sigma QUBO scored 0.1140 solve@1 / KL 2.4737 and failed its gate;
   min ||Y_clean - W_hat X_tilde|| scored 0.1605 solve@1 / KL 2.3322 / `game_agree`
   0.1440 and collapsed the KL gap to the factored init from 0.235 to 0.039 nats
   (§5.4.2, §5.5). It still missed the 0.18 solve@1 bar by 2.0 pp.
4. **Cosine decay removes the late-run KL blowup** and sets the clean 1.04 bpw
   yardstick: KL 0.5993 and 1495 Elo greedy, against 0.9487 and 1245 for the same arm
   with constant LR whose KL turned upward after step 3,600 (§5.8, §5.9).

What is *not* claimed: that no full-rank 1-bit schedule works. The evidence is that
the standard STE with any latent scale reachable by Adam bursts all reachable signs
by step 200 and loses the seed, and that a sign-specific optimiser was not tried. If
full-rank STE is revisited, that is the experiment to run, and §5.6-§5.7 define the
gate it must pass.




