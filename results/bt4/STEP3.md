# BT4 at 1.04 bpw — Step 3: what rating does it actually play at?

Step 2 ended with the student finding the teacher's move on **76.0 %** of 2,000
held-out puzzles (`qubo_act`, 1.0415 bpw). That is a one-ply agreement number.
It says nothing about a 60-move game, where a 24 % per-move error rate compounds
and the value head matters. Step 3 measures the difference: games, not puzzles.

Entry condition (declared before Step 3 was built): either arm ≥ ~0.70–0.75
`solve@1`. `qubo_act` reached **0.7600**, so the student measured here is the
`qubo_act` arm, seed 0, 2,000 steps, batch 64.

## 3.0 Harness (`harness/bt4/elo_match.py`)

Both players reduce to `move(board) -> chess.Move`:

* **the model** — greedy on the legal-masked policy, temperature 1, exactly the
  decision the puzzle gate scored. fp16 on the GPU: this is the deployed
  object, not a float32 readout. Castling and underpromotion are encoded
  non-literally by the Leela vocabulary, so the index → move inverse is built
  from `board.legal_moves` per position rather than by decoding `vocab.moves[i]`.
* **Stockfish 19** (`tools/stockfish/...universal.exe`, `id name Stockfish 19`)
  with `UCI_LimitStrength=true` and `UCI_Elo`, the anchor the result is
  reported against.

Design choices that keep the number honest:

* Colours alternate every game; 8 short sound openings, seeded, so neither side
  replays one line 32 times.
* Rating is **not** read off one rung. Games at four opponent settings are
  pooled and a grid fit of `result = 1/(1+10^((opp−R)/400))` gives the estimate.
* Uncertainty is a **2,000-sample bootstrap over games**, not an SSE band: with
  a handful of decisive games an SSE band is meaningless while the resampled
  spread is not.
* Ply cap 140, and a capped game is **adjudicated** by the rung's Stockfish at
  fixed depth (|cp| > 200 decisive, else draw).

### The control, and the bug it caught

Before trusting the harness on the student, it was run with Stockfish in the
model's seat at a known `UCI_Elo=1500` (`results/bt4/step3_elo_sf_control.json`,
32 games, 30.2 s):

| fit | value |
|---|---|
| recovered rating | **1610** |
| 95 % bootstrap CI | [1445, 1780] |
| anchor | 1500 ✓ inside CI |
| decisive games | 30 / 32 |

The first version of this harness scored the control at **1160**. Cause: a game
that reaches the ply cap returns `"*"` from `board.result()`, and `"*"` was
being mapped to a **loss** instead of a draw — six adjudicated draws were
counted as six defeats. That is the failure mode worth remembering for this
procedure: the rating never came out implausible, it came out *plausibly wrong*,
and only a control with a known answer showed it. The fix is `adjudicate()`
above plus `"*" → 0.5` in the scorer.

## 3.1 The student, and the teacher baseline

Step 2's trainer wrote only metrics, so the `qubo_act` arm was re-run under tag
`full_qubo_act_ckpt` with `--save` (signs int8, scales fp32). It reproduced the
original **exactly** — 0.2350 / 0.5260 / 0.6455 / 0.6905 / 0.6970 / 0.7345 /
0.7355 / 0.7455 / 0.7535 / 0.7550 / **0.7600** at steps 0…2000, same agree and KL
(0.6130 / 0.6251), 1854.9 s, 7.64 GiB — so it is that student, not a proxy.
Checkpoint: `results/bt4/step2_weights_full_qubo_act_ckpt.pt`.

The same ladder was run for the fp16 teacher, because a student rating alone
says nothing about what 1 bit cost.

| player | 1320 | 1600 | 1900 | 2200 | 2500 | fit | 95 % CI |
|---|---|---|---|---|---|---|---|
| fp16 teacher | 8/8 | 8/8 | 7/8 | 8/8 | 4.5/8 | **2650** | [2495, 2950] |
| 1.04 bpw student | 1/8 | 0/8 | 0/8 | 0/8 | 0/8 | — | — |

`results/bt4/step3_elo_qubo_act.json`, `…_fp16_teacher_high.json`,
`…_qubo_act_1320.json`.

The 5-rung fit for the student bottomed out at 970 [200, 1220] — but 970 is an
extrapolation below the lowest `UCI_Elo` anchor Stockfish allows (1320). So the
student was measured where it can actually be anchored, 32 games at 1320
(`+2 =6 −24`, six adjudicated draws):

| student, 32 games vs `UCI_Elo` 1320 | value |
|---|---|
| score | 0.1562 |
| rating | **1025** |
| 95 % bootstrap CI | [850, 1145] |
| decisive games | 26 / 32 |
| model time per move | 0.027 s |

**The 1.0415 bpw student plays about 1025 Elo; its fp16 teacher plays about
2650. One bit costs ~1600 Elo, while costing only 18 pp of one-ply puzzle
agreement (0.7600 vs 0.9440).**

### Why this is not a harness bug

A 1600-Elo collapse on a 76 % agreement number needs ruling out as a wiring
fault before it is a result. Three checks, same code path for both players:

1. **The teacher wins through the same `ModelPlayer`** (16/16 at 1320+1600), so
   the player, the plane builder, the dtype and the greedy rule are not the
   problem.
2. **The student still finds the gold move through the in-game path**:
   `_diag_play.py` drives `ModelPlayer` over 200 test puzzles and gets
   **0.730** agreement (the gate itself is 0.7600 in fp32 batch-64 — the small
drop is the fp16 batch-1 path). It is being asked the same question.
3. **It is not one degenerate move**: raw top-5 on the start position are
   promotion-garbage indices, and the masked best is `a2a3` — a planless move,
   which is what a policy trained only on tactical puzzles should produce.

So the gap is behaviour, not plumbing.

### What the gap actually measures (and what it does not)

`2650 vs 1025` conflates two things, and only one of them is quantisation:

* **Greedy, search-free play.** Controlled: the teacher is also greedy with no
  search and still reaches 2650, so search is not the explanation.
* **Training distribution.** *Not* controlled. The student trained on 60,000
  Lichess **puzzle** positions — tactical middlegames with one ply of history.
  Openings, quiet manoeuvring and endgames are out of distribution for it, and
  a game walks straight out of the pool after a few plies. The fp16 teacher was
  trained on real games and does not have that handicap.

That means the honest headline is narrower than "1 bit costs 1600 Elo":
**distilling onto puzzle positions for 128k steps makes a 76 %-agreement policy
whose play is ~1025 Elo**, and this experiment cannot yet say how much of the
missing 1600 is the bit and how much is the pool. Separating them needs a
control the current data cannot supply — fine-tuning the *fp16* model with the
same loss on the same 128k positions, which bounds what that pool can teach at
any precision. That is the next experiment, not a conclusion.

One more caveat on absolutes: `UCI_LimitStrength` Elo anchors are calibrated at
normal time controls. At 10 ms/move the Stockfish opponent is weaker than its
label, so the absolute numbers are indicative and the teacher-vs-student
comparison — identical ladder, openings, colours and clocks — is the solid part.

## 3.2 What this does to the Step 2 verdict

Step 2's S1/S2 passes stand as measured, but they do not support "the binary
model is nearly as strong". `solve@5 = 0.9735` is the number that looks strongest
and is the least predictive of play: the right move being somewhere in the top
five is worth almost nothing when the player must spend its one move on it.
`agree = 0.613` and the 24 % per-move miss rate were the honest predictors, and
in play they compound over ~27 of the model's own moves per game.

The practical reading for the programme: at 1.04 bpw the body is trainable and
preserves the teacher's ranking far better than it preserves its *choices*, so
any further spend should target per-move accuracy (game-position distillation,
more steps, or search at inference) rather than KL or `solve@5`.
