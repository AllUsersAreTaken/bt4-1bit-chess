# BT4 at 1 bit per weight

[BT4](https://lczero.org) is a strong Leela-style chess network — 1024×15×32,
SWA. As fp16 it plays about **2650 Elo** on this measurement.

Compress it to **1.04 bits per weight** and it plays **1495 Elo**. One bit costs
roughly **1,600 Elo while costing only 18 percentage points of one-ply puzzle
agreement** (0.7600 against 0.9440). That gap between the cheap proxy and the
thing you actually care about is what this repository is about.

Every number here came from a run on one machine — an RTX 4060, 8 GB, 115 W —
with one seed, one calibration set and one bit rate. Nothing is extrapolated to
another GPU.

## The three results that matter

**1. The proxy metric is nearly useless here.** One-ply move agreement sits at
0.76 and reads as "mostly intact". The engine is 1,600 Elo weaker. If you had
tuned the compression against agreement, you would have shipped something that
cannot play.

**2. Rank, not bits, is where the loss starts.** At the same 1.04 bpw budget,
exact SVD to rank `d/2` discards **19.5 %** of each matrix (mean Frobenius; up to
31.9 %) and drops the model from 0.9440 to **0.3110** solve@1 and from 1.0 to
**0.2179** `game_agree` — *at 16 bits per factor*. The bit is not the expensive
part of the budget; choosing a subspace the teacher does not live in is. QAT
recovers a great deal inside it (0.3110 → 0.6855), which is the interesting part.

**3. A sign optimiser that did nothing, and a metric that did not notice.** The
full-rank `bit1` arm never moved **a single sign** — exact zero sign flips over
training — and still passed its one-ply gate. That is recorded as an amendment in
[`results/bt4/STEP5.md`](results/bt4/STEP5.md) §5.4 rather than quietly dropped,
because it is the sharpest example here of a gate that cannot see the failure it
is supposed to catch.

## Ratings

Elo is measured through Stockfish's `UCI_LimitStrength` / `UCI_Elo` anchors —
32 games per rung, bootstrap CIs. Stockfish will not anchor below 1320, so the
student was measured at 1320 where it can actually be anchored.

| player | fit | 95 % CI |
|---|---|---|
| fp16 teacher | **2650** | [2495, 2950] |
| 1.04 bpw, low-rank (`qubo_act`) | **1025** | [850, 1145] |
| 1.04 bpw, cosine-LR yardstick, greedy | **1495** | — |

The cosine LR decay is worth 250 Elo on its own (1495 against 1245 for the same
arm with constant LR), because it removes the late-run KL blowup: KL 0.5993
against 0.9487.

## Layout

```
harness/bt4/     16 files: bit-path forward, QUBO sign search, QAT, Elo harness
results/bt4/     STEP0..STEP5.md - the write-up, step by step, with the gates
                 and amendments inline - plus the JSON each claim cites
```

The `STEP*.md` files are the actual record: gates were frozen before arms were
scored, and every amendment is dated and appended rather than edited in.
[`STEP0.md`](results/bt4/STEP0.md) §0.6 has the frozen gates;
[`STEP5.md`](results/bt4/STEP5.md) §5.10 is the close-out.

Not in the repository: the 3.3 GB of inputs. BT4 weights
(`BT4-1024x15x32h-swa-6147500.pb.gz`) come from the Leela project; the training
pool is 80 % real games and 20 % puzzles from
[database.lichess.org](https://database.lichess.org)
(`lichess_2013-01.pgn.zst`, `lichess_db_puzzle.csv.zst`). `data/` is the place to
put them.

## What is not claimed

- **Not** that no full-rank 1-bit schedule works. The evidence is that standard
  STE with any reachable latent scale bursts all reachable signs by step 200 and
  loses the seed. A sign-specific optimiser was not tried; §5.6–5.7 define the
  gate it would have to pass.
- **Not** that the rank-`d/2` family is capped at 0.47 `game_agree`. Exact SVD is
  the best *weight-space* approximation, not a behavioural ceiling — training
  beats it, which is what a trained rank-r model is supposed to do.
- **Not** that 1495 Elo is the ceiling at 1.04 bpw. That is one arm with a cosine
  schedule, measured greedy.

One seed, one bit rate, no repeats. The CI on the 1025 is a bootstrap over 32
games and is honest about the game count, not about the seed count.

## Licence

MIT. See [LICENSE](LICENSE).
