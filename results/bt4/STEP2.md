# BT4 at 1.0 bpw — Step 2: distillation of the binary body

Scope 2 (149.91M weights, every 2-D linear in the 15 blocks) at **1.0415 bpw**.
Student = the Step 1 arm's own LittleBit factors, run factored with
straight-through signs. Teacher = the fp16 BT4, resident on the same card.
Positions = 60,000 Lichess puzzles disjoint from the 2,512 eval puzzles
(`data/bt4/train.jsonl`, FEN overlap **0** by construction).

## 2.0 What had to be built

Step 1 wrote `path.effective_weight()` into each module, so its activations were
the real binary stream but the factors were gone — and rebuilding 105 dense
150M-element matrices per step is ~150 TFLOP, not a training loop. Hence:

* `harness/bt4/bitpath.py` — `BitPath` runs `y = ((x⊙g)·sgn(V))⊙l·sgn(U)ᵀ⊙h`
  forwards. Storage is unchanged: `(d_in+d_out)·r` sign bits + fp16 scales.
  `sgn` uses the plain STE (identity backward); LittleBit's `SmoothSign` at
  k=100 has gradient ≈ 0 away from zero, which would freeze a trained sign
  pattern rather than let it flip.
* `_sdpa_forward` — explicit `F.scaled_dot_product_attention` replacing
  `nn.MultiheadAttention`'s fused call, so the **packed** QKV projection and
  `out_proj` both run as factored paths. The fused kernel reads
  `in_proj_weight` directly, which is the sole reason Step 1 needed dense
  weights (and why the `out_proj` Σ hook never fired).

### Forward equivalence (frozen check, `results/bt4/step2_bitpath_check.json`)

Dense write of the arm's effective weight vs the factored module built from the
same factors, both fp32, 8 calib positions:

| quantity | value |
|---|---|
| max abs Δ policy logits | 6.91e-06 |
| max rel Δ policy logits | 1.96e-06 |
| max abs Δ `in_proj` weight | 0.0 |
| bits / bpw | 156,127,920 / **1.0415** |

Identical bpw to Step 1, so the student is the same 1.04 bpw object.

Three bugs this check caught, all worth recording:

1. `_sdpa_forward` first omitted `mha.in_proj_bias` → 0.52 max logit Δ.
2. A 3-D `(B*H,64,64)` mask is not broadcastable against a 4-D SDPA query;
   the bias must stay `(B,H,64,64)`.
3. The dense reference had been written in fp16 (`quantise()` writes fp16 on
   purpose) while the factored side was fp32 — the comparison has to be fp32
   vs fp32.

## 2.1 A reproducibility finding that changes how Step 1 reads

`dual_svid(init="itq")` starts `joint_itq` from a **random orthogonal R**, and
nothing seeded it. The same `itq` quantisation therefore produces a different
sign pattern on every run, and the Step 1 table carries that noise:

| run | itq zero-shot solve@1 |
|---|---|
| Step 1 (unseeded) | 0.1375 |
| this turn, unseeded draw A | 0.2140 |
| this turn, `seed=0` | 0.1185 |

Spread ≈ 10 pp at batch-of-one draws — the same order as the +5.4 pp gap Step 1
attributed to `qubo_act`. **Step 1's arm ranking is therefore not established by
a single unseeded run per arm**; `qubo_act` vs `itq` needs ≥3 seeds before the
lead means anything. `qat.py` now seeds (`--seed`, recorded in `head.seed`).

## 2.2 VRAM: the 8 GB card is the binding constraint, not the math

The 200-step pilot at batch 128 peaked at **10.06 GiB on an 8188 MiB card** —
WDDM silently spilled to host memory and step time was 5.07 s (≈25 pos/s, while
the same forward in isolation runs at 1048 pos/s). At batch 64 the same loop
runs **0.75–0.85 s/step** with `nvidia-smi` flat at 7908 MiB and no spill.

Consequence for any future debugging here: **a slow BT4 training step is a
memory symptom first.** Reading `peak_gib` from the run JSON is cheaper than
profiling, and a run that "gets slower over time" is usually fragmentation.

## 2.3 The pilot (itq, seed 0, batch 128, 200 steps, 25.6k positions)

`results/bt4/step2_qat_pilot_itq.json`, 1070 s wall, 10.06 GiB peak (spilled):

| step | solve@1 | solve@5 | agree | KL |
|---|---|---|---|---|
| 0 | 0.1185 | 0.3740 | 0.1025 | 2.4663 |
| 50 | 0.2320 | 0.7025 | 0.2060 | 1.9564 |
| 100 | 0.4255 | 0.8710 | 0.3220 | 1.3286 |
| 150 | 0.5040 | 0.9095 | 0.3930 | 1.1212 |
| 200 | **0.5320** | **0.9240** | 0.4140 | **1.0320** |

Zero-shot 0.1185 → 0.5320 solve@1 in 200 steps, KL −58 %, no NaN. The student
climbs out of the Step 1 collapse with policy KL + WDL CE + last-block hidden
MSE, which is the result the whole programme needed: **1.0415 bpw is trainable
back toward the teacher**, it is not a dead quantisation.

`solve@5` at 0.924 already sits inside 7 pp of the fp16 teacher's 0.9980 while
`agree` is only 0.414 — the binary model recovers the *right move* far more
often than it reproduces the teacher's exact ranking. Expected under 1 bit, and
the reason `solve@5` is the more informative curve early on.

## 2.4 Full runs (launched 22:05, batch 64, 2,000 steps = 128k positions each)

```
python harness/bt4/qat.py --arm itq      --steps 2000 --batch 64 --every 200 --seed 0 --tag full_itq
python harness/bt4/qat.py --arm qubo_act --steps 2000 --batch 64 --every 200 --seed 0 --tag full_qubo_act
```

Readouts land every 200 steps in `results/bt4/step2_full_{itq,qubo_act}.log`
and, on completion, `results/bt4/step2_qat_full_*.json`.

### Readout criteria for those runs (declared 22:15, before any full-run readout past step 400 was seen)

| gate | condition |
|---|---|
| S1 | `solve@1` ≥ 0.75 (≥ 79 % of the fp16 teacher's 0.9440) |
| S2 | `solve@5` ≥ 0.95 |
| S3 | top-1 agreement with fp16 ≥ 0.70 |
| S4 | legal-move KL ≤ 0.40 |
| S5 | `qubo_act` ≥ `itq` on final `solve@1` — the Step 1 lead must survive distillation to count as a representation advantage |

S1–S4 are absolute; S5 is the only comparative claim, and per §2.1 it needs
replication across seeds before it is a finding rather than a draw.

### itq trajectory

| step | solve@1 | solve@5 | agree | KL | s/step |
|---|---|---|---|---|---|
| 0 | 0.1185 | 0.3740 | 0.1025 | 2.4663 | — |
| 200 | 0.5025 | 0.9180 | 0.3830 | 1.1093 | 0.75 |
| 400 | 0.5670 | 0.9360 | 0.4485 | 0.9452 | 0.82 |
| 600 | 0.6000 | 0.9490 | 0.4730 | 0.8667 | 0.85 |
| 800 | 0.6020 | 0.9445 | 0.4710 | 0.8712 | 0.85 |
| 1000 | 0.6110 | 0.9500 | 0.4745 | 0.8402 | 0.85 |
| 1200 | 0.6515 | 0.9510 | 0.5185 | 0.8539 | 0.85 |
| 1400 | 0.6575 | 0.9560 | 0.5225 | 0.8193 | 0.85 |
| 1600 | 0.6585 | 0.9555 | 0.5170 | 0.8308 | 0.85 |
| 1800 | 0.6640 | 0.9515 | 0.5325 | 0.8620 | 0.85 |
| 2000 | **0.6730** | 0.9470 | 0.5415 | 0.8865 | 0.89 |

**The curve is stepwise, not saturating.** 800 → 1000 gained +0.9 pp and looked
flat, then 1000 → 1200 gained +4.1 pp. That is the signature of discrete sign
flips under the STE: the scales move smoothly, but accuracy only moves when a
latent crosses zero, so single flat intervals say nothing about convergence.
(Recorded because the first reading of the step-1000 point *was* "saturated at
0.61" — the step-1200 readout falsified it, and an earlier write-up of this
document said so.)

Consequence for reading any of these curves: fit or judge only across ≥600-step
spans, never from one interval, or a plateau will be mistaken for a ceiling.

## 2.5 Final results

Both receipts landed, `ITQ_EXIT=0` and `QUBO_EXIT=0`, both at **7.64 GiB peak**
(no host spill, which is why both ran the full 2,000 steps in ~30 min):

* `results/bt4/step2_qat_full_itq.json` — 1787.0 s, 0.89 s/step
* `results/bt4/step2_qat_full_qubo_act.json` — 1848.3 s, 0.92 s/step

### qubo_act trajectory

| step | solve@1 | solve@5 | agree | KL |
|---|---|---|---|---|
| 0 | 0.2350 | 0.4790 | 0.2025 | 2.2931 |
| 200 | 0.5260 | 0.9200 | 0.4070 | 1.0882 |
| 400 | 0.6455 | 0.9555 | 0.5115 | 0.8539 |
| 600 | 0.6905 | 0.9610 | 0.5470 | 0.7338 |
| 800 | 0.6970 | 0.9595 | 0.5595 | 0.7202 |
| 1000 | 0.7345 | 0.9740 | 0.5940 | 0.6567 |
| 1200 | 0.7355 | 0.9665 | 0.5915 | 0.6337 |
| 1400 | 0.7455 | 0.9760 | 0.5955 | 0.6106 |
| 1600 | 0.7535 | 0.9730 | 0.6135 | 0.5888 |
| 1800 | 0.7550 | 0.9755 | 0.6165 | 0.5930 |
| 2000 | **0.7600** | **0.9735** | 0.6130 | 0.6251 |

`qubo_act` reaches the fp16 teacher's `solve@1` in 400 steps that `itq` needed
1,200 to reach, i.e. it does not merely finish higher, it starts the run a
third of the way up (zero-shot 0.2350 vs 0.1185). Same stepwise shape, same
late deceleration: 1200 → 2000 is only +2.5 pp over 800 steps.

### Head to head at 2,000 steps (both batch 64, seed 0, 128k positions)

| arm | zero-shot solve@1 | final solve@1 | final solve@5 | agree | KL |
|---|---|---|---|---|---|
| `itq` | 0.1185 | 0.6730 | 0.9470 | 0.5415 | 0.8865 |
| `qubo_act` | 0.2350 | **0.7600** | **0.9735** | **0.6130** | **0.6251** |
| fp16 teacher | — | 0.9440 | 0.9980 | 1.0 | 0 |

The 1.0415 bpw student ends at **80.5 % of the teacher's solve@1** and
**97.6 % of its solve@5**, with an 8.7 pp `solve@1` lead for `qubo_act`.

Also note `itq`'s `solve@5` is **non-monotone** (0.951 at 1200, 0.956 at 1400,
0.947 at 2000) while `qubo_act`'s holds ≥ 0.9595 from step 400 on. Single sign
flips reshuffle the tail of the ranking both ways; only the ≥600-step trend is
meaningful.

### Gate verdicts (criteria frozen in §2.4 before step 400 was seen)

| gate | condition | `itq` | `qubo_act` | verdict |
|---|---|---|---|---|
| S1 | solve@1 ≥ 0.75 | 0.6730 ✗ | **0.7600 ✓** | met by `qubo_act` |
| S2 | solve@5 ≥ 0.95 | 0.9470 ✗ | 0.9735 ✓ | met by `qubo_act` |
| S3 | top-1 agree ≥ 0.70 | 0.5415 ✗ | 0.6130 ✗ | **not met** |
| S4 | legal-move KL ≤ 0.40 | 0.8865 ✗ | 0.6251 ✗ | **not met** |
| S5 | `qubo_act` ≥ `itq` | — | +8.7 pp | passes on seed 0 only |

S1 and S2 pass, S3 and S4 do not. The pattern is internally consistent and is
the interesting part: the student picks the teacher's move 76 % of the time
while matching its full ranking only 61 % of the time and holding KL 0.63.
**One bit preserves the decision and loses the distribution.** Any downstream
use that only reads the argmax (play, puzzle solve, Elo) should be far less
damaged than the KL number suggests; anything that needs calibrated move
probabilities or a value curve is where the 1-bit body actually costs.
S5 passes but per §2.1 one seed per arm does not establish it — it needs ≥3
seeds before it is a representation claim rather than a lucky draw.

## 2.6 Gap found after the runs: no checkpoint was written

`qat.py` saved only the metrics JSON. The trained factors lived in the process,
so the 0.7600 student could not be replayed — and Step 3 (an Elo match) needs
exactly that object. `--save` now writes
`results/bt4/step2_weights_<tag>.pt` with signs as **int8** (exact, 1 byte per
latent) and scales fp32, plus the run head; `load_factors()` installs them onto
a fresh BT4 so the trained student is reconstructible without re-training.

Because the evidence must come from a real run, the arm was re-run under a
separate tag (`full_qubo_act_ckpt`) rather than assumed to reproduce: training
is seeded (`--seed 0`, data order from `default_rng(0)`) so the two runs should
agree, and the comparison doubles as a seed-0 reproducibility check.

It reproduced **exactly**: identical `solve@1` at all eleven readouts
(0.2350 / 0.5260 / 0.6455 / 0.6905 / 0.6970 / 0.7345 / 0.7355 / 0.7455 /
0.7535 / 0.7550 / 0.7600), identical agree 0.6130 and KL 0.6251, 1854.9 s vs 1848.3 s,
7.64 GiB both. So the `--save` checkpoint is the student reported in §2.5, and
seed 0 is bit-reproducible on this box — worth knowing before any comparison
across seeds is attempted.

Saved factors: `results/bt4/step2_weights_full_qubo_act_ckpt.pt`
(int8 signs + fp32 scales + the run head). That is the object Step 3 plays
with; see `STEP3.md`.
