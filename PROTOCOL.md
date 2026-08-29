# Omnipex Value Test — Protocol

The purpose of this experiment is to **kill or confirm one hypothesis** before any
more product is built. It is a signal-seeking test, not a publication — we are
looking for a large, obvious effect. If the effect is small, the wedge is weak and
we want to know now.

## Hypothesis (the narrow one)

> When several AI-pipeline changes land in one commit and a metric regresses,
> Omnipex's attribution output lets an engineer identify the culprit **faster**,
> with **higher accuracy**, and **higher confidence** than today's manual
> revert-and-rerun workflow — and the advantage **grows with difficulty**.

We are NOT testing "is evaluation hard" or "do people like DeepEval." We are
testing multi-change attribution specifically.

## What we measure (per scenario, per participant)

- **Time-to-culprit (adjusted):** wall-clock to commit an answer, plus a simulated
  CI cost for each manual re-run in the control arm (default 180s/run — set it to
  your real eval latency with `--penalty`). This is the honest cost: today each
  isolation is a CI cycle, not a mouse click.
- **Accuracy:** did they name the true culprit (or correctly say "noise")?
- **Confidence:** self-reported 1–5.
- **Re-runs:** how many manual isolations the control arm needed.

## Design: within-subjects, counterbalanced

Each participant does **every** scenario, but each scenario in **one** arm only
(never both — that would leak the answer). Arms alternate across scenarios, and
the starting arm is set per participant so that, across participants, each scenario
is seen in both arms roughly equally. `valuetest` handles this automatically
(`--arm auto`). Scenario order is shuffled per participant to spread learning
effects.

Rationale: with the handful of engineers you can realistically recruit,
within-subjects gives far more statistical power than splitting people into two
groups.

## The arms

- **Control (today):** participant sees the changed params and the before/after
  eval numbers, then investigates by isolating one change at a time and re-running
  (raw means only — no significance verdict, just like eyeballing CI output). Each
  re-run adds the CI-time penalty.
- **Treatment (Omnipex):** participant sees the `blame` attribution — named
  culprit, per-change isolated effect with REAL/NOISE verdicts, masking notes, or
  "no real regression" — and reads off the answer.

## Scenarios (difficulty gradient — see `scenarios.py`)

1. `s1` easy — single embedder swap (intuition should suffice; expect ~no gap).
2. `s2` medium — embedder swap + innocent top_k bump.
3. `s3` hard — a drop **masked** by an apparent improvement.
4. `s4` hard — the intuitive suspect (model swap) is wrong; chunking dominates.
5. `s5` medium — a counterintuitive recall↑/precision↓ tradeoff.
6. `s6` hard — the "regression" is **pure noise**; correct answer is "nothing."

`s1` is the control condition for the experiment itself: if Omnipex shows no
advantage there but a large one on `s3`/`s4`/`s6`, the value gradient is real.
`s6` tests the most expensive real-world failure: the wild-goose chase.

## Procedure (facilitator)

1. `python -m valuetest.cli check` — confirm scenarios are consistent.
2. Recruit an AI/ML engineer who has shipped a RAG or agent pipeline.
3. Neutral framing only: *"I'm comparing two ways to debug a regression. Work as
   you normally would."* **Do not** say which arm is "the product." **Do not**
   coach, hint, or react during a scenario.
4. `python -m valuetest.cli run --participant <initials> --penalty <your CI secs>`
5. After all participants: `python -m valuetest.cli analyze`.

Target **8–12 participants** (≈50–70 observations/arm). A real effect will be
visible well before that; stop early if it's overwhelming in either direction.

## What a simulated backend does and does not prove

Scores come from a calibrated simulator, not real evals. This experiment therefore
proves **whether the *attribution workflow* changes how engineers debug** — the
human/interaction claim. It does **not** prove Omnipex attributes correctly on real
data; that is the next step (live DeepEval), and it is only worth doing if this
test passes.

## Decision rule (commit to this before collecting data)

- **PASS →** build the live DeepEval backend next, if:
  - treatment median adjusted time ≤ **0.5×** control, AND
  - treatment accuracy ≥ control accuracy, AND
  - the time/accuracy gap on **hard** scenarios is clearly larger than on `s1`.
- **PIVOT →** if control engineers are already fast *and* accurate (the manual
  workflow isn't painful enough), or the gap doesn't widen with difficulty. The
  attribution wedge is weaker than believed; revisit positioning.
- **FAIL →** no meaningful time or accuracy difference. The hypothesis is wrong;
  do not build more engine.

## Threats to validity (watch for these)

- **Simulated re-runs feel free.** Mitigated by the CI penalty; set it honestly.
- **Expert intuition** may solve easy scenarios instantly — expected, and why the
  gradient matters more than the average.
- **Demand characteristics** — participants guessing which arm you're rooting for.
  Mitigate with neutral framing and by not revealing the product name until after.
- **Small N** — treat this as directional signal, not a p-value. Look for a large
  effect or none.
