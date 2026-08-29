# OmniPex

**Regression CI for AI pipelines.** You change a component — a model, a prompt, a
`top_k` — and OmniPex tells you, *before you ship*, what downstream behaviour
actually broke, what needs rebuilding, and whether an observed score change is
**real** or just LLM noise.

This is **Stage 1: the intelligence layer** — built, hardened, and awaiting
validation. It answers one question cheaply: *do AI engineers find this
indispensable?* Everything beyond it (auto-extraction, `doctor`, MCP
distribution, the workspace) is sequenced behind that answer in
[VISION.md](VISION.md).

## Why this and not "another AI IDE"

Cursor, Copilot, and Windsurf/Codemaps already map **code** structure. LangSmith,
Braintrust, and Phoenix already run **evals**. Nobody models the **invalidation
graph**: *this specific change made these specific downstream evals stale — here
is the blast radius and the attributed cause.* That is the wedge.

> LangSmith tells you your eval scores. omnipex tells you which change caused them
> to move — before you ship it.

## Install & run (zero dependencies, Python 3.10+)

```bash
pip install -e .        # provides the `omnipex` command
# (or run everything below as `python -m omnipex.cli ...` with no install)

# inspect a pipeline graph
omnipex show examples/rag_docs_qa/manifest.json

# instant check: does the config match what was actually built? (no eval cost)
omnipex doctor examples/rag_docs_qa/manifest_v2.json

# change-impact analysis: what's the blast radius, what's real vs noise?
omnipex diff examples/rag_docs_qa/manifest.json examples/rag_docs_qa/manifest_v2.json

# the headline: bisect the diff and NAME the change that caused each regression
omnipex blame examples/rag_docs_qa/manifest.json examples/rag_docs_qa/manifest_v2.json

# the same engine in a browser: pipeline graph, doctor, diff, and blame views
omnipex ui

# the same engine inside Claude Code / Cursor (MCP server over stdio)
omnipex mcp --dir .

# tests (pins the domain knowledge that is the actual IP)
python tests/test_invalidation.py
```

`diff` uses CI exit codes, so it works as a merge gate today:
`0` = clean, `1` = real regression detected (block the merge), `2` = input error
(malformed manifest, unknown rule-pack, unsupported structural change — all with
precise messages, never a traceback).

The example diff swaps the embedder to a weaker model **and** bumps `top_k` — a
change that *looks* like an upgrade. `diff` reports the blast radius and flags the
real regressions; `blame` then **isolates each change** and names the culprit:
the embedder swap tanked `faithfulness` and `answer_correctness`, while the
`top_k` bump was innocent (and actually *masked* part of the correctness drop by
improving it). That "here's the culprit, and here's why" output — not blast-radius
visualization — is the core claim.

### `doctor` — config vs. reality (instant, free)

The invalidation engine says an artifact *should* be rebuilt. `doctor` verifies it
*was*. Nodes that declare `produces_artifact` can also declare a `stamp` — a small
JSON file your build script writes, recording what the artifact was really built
with:

```json
{ "artifact": "vector_index", "built_at": "2026-07-14T09:12:00Z",
  "params": { "chunker.size": 512, "embedder.model": "text-embedding-3-small" } }
```

`doctor` compares stamp against manifest and reports drift — the classic silent
RAG killer (index built with one embedder, queried with another; nothing errors,
quality quietly degrades). **Which parameters matter is derived from the
rule-pack**, not hardcoded: a param is checked when changing it would propagate
`RECOMPUTE_REQUIRED` to that artifact. New rule-packs get consistency checking for
free. Nodes may also declare `requires_env` for a static missing-key check.

Exit codes match `diff` (`1` on a critical inconsistency), so it gates CI too — in
seconds, with no judge tokens spent.

> **Consistency is not health.** A fully consistent pipeline can still answer
> badly. `doctor` never claims otherwise, and the output says so explicitly.

### `diff` vs `blame` (suspects vs. culprit)

`diff` narrows to the *suspect set* (every change whose blast radius reaches a
regressed metric). `blame` does **change-set bisection**: it applies each suspect
alone to the baseline, re-runs only the at-risk metric, and ranks changes by their
*isolated* real effect to name the actual culprit. The invalidation graph is what
makes this cheap — blind bisection is `|changes| x |metrics|` runs; the graph
prunes to only the pairs that can interact.

## Use it from Claude Code / Cursor (MCP)

OmniPex ships an MCP server, so your assistant can answer *"which change broke
faithfulness?"* by calling the engine directly — no context switch, no new app to
learn. Add this to your host's MCP config (full example:
[examples/mcp_config.json](examples/mcp_config.json)):

```json
{ "mcpServers": { "omnipex": { "command": "omnipex", "args": ["mcp", "--dir", "."] } } }
```

Four tools are exposed — `omnipex_show`, `omnipex_doctor`, `omnipex_diff`,
`omnipex_blame` — mapping one-to-one onto the CLI commands. The transport is
newline-delimited JSON-RPC 2.0 over stdio, implemented with the standard library
(no MCP SDK, no dependencies). Paths resolve inside `--dir` and cannot escape it.

## Architecture (the boundary that matters)

```
omnipex/
  core/            # GENERIC. knows nothing about RAG. never rewritten per vertical.
    schema.py        Node / Edge / EffectType / Pipeline (validated DAG) / Change
    errors.py        OmnipexError hierarchy (clean messages at every entry point)
    rulepack.py      RulePack plugin interface + registry
    invalidation.py  propagate a change -> effects; full analyze()
    attribution.py   change-set bisection: suspects -> culprit (graph-pruned)
    consistency.py   declared config vs. build stamps (rule-pack derived)
  rulepacks/
    rag.py         # THE IP. hand-authored invalidation matrix for RAG, as data.
  eval/
    stats.py         paired bootstrap + noise band (the hard "REAL vs noise" call)
    runner.py        eval harness (real) + SIMULATED score source (swap for ragas)
  api.py           # service layer: every capability as a JSON-ready dict —
                   # the single source of truth for CLI, GUI, and future MCP/API
  ui/              # local GUI (stdlib http server + one HTML file); a VIEW over
                   # api.py — adds no capability the CLI lacks, by design
  mcp/             # MCP server (stdlib JSON-RPC over stdio); another VIEW
  manifest.py      # Phase 0 loader (validating); Phase 2: trace reconstruction
valuetest/         # the Phase 1 experiment (PROTOCOL.md, RECRUITING.md) — not shipped
timing/            # standalone DeepEval stopwatch for the CI penalty — not shipped
tests/             # zero-dependency suite; also pytest-compatible
```

**Design for generality, build for specificity:** the core schema already
describes agents / voice / multimodal. Only the RAG rule-pack is implemented.
Adding a vertical = authoring a new rule-pack, not touching `core/`.

## What is real vs. simulated here

| Piece | Status |
|---|---|
| Invalidation engine + RAG matrix | **real** |
| Blast-radius / recompute / proven-safe logic | **real** |
| Paired bootstrap + noise bands | **real, production-shaped** |
| Family-wise false-discovery control | **real** (measured: cuts per-run false alarms from 11.3%) |
| Eval *score source* | **simulated** (calibrated stand-in; swap for ragas/DeepEval/LLM-judge) |
| Graph source | hand-written manifest (Phase 0); trace-extraction is Phase 2 |

## The value test (the actual next milestone)

`valuetest/` is a **measurable experiment**, not part of the product: does
Omnipex's attribution reduce time-to-culprit and raise accuracy/confidence vs
today's manual revert-and-rerun? See [PROTOCOL.md](PROTOCOL.md) for the full
design (within-subjects, counterbalanced, explicit decision rule).

```bash
python -m valuetest.cli check                       # scenarios agree with the engine
python -m valuetest.cli preview s3_masked_correctness --arm treatment
python -m valuetest.cli run --participant AB --penalty 180   # live session
python -m valuetest.cli analyze                     # aggregate results
```

Six difficulty-graded scenarios include a masked regression, a case where the
intuitive suspect is wrong, and a pure-noise trap (correct answer: "nothing").
The point is to find *where* Omnipex starts to matter — the value should
concentrate on the hard cases, not the easy ones.

## Roadmap & kill-criteria

[VISION.md](VISION.md) is the canonical strategy document: the destination (an
all-in-one AI engineering workspace), why we enter at the intelligence layer, the
seven staged roadmap with **kill-criteria** at each gate, the business model, and
what is deliberately not being built.

**Current gate — Stage 2, validation.** Put this in front of 8–12 working AI
engineers and measure time-to-culprit against their manual workflow. Stage 3
(live DeepEval backend, `omnipex doctor`, trace auto-extraction) is gated on a
PASS. Nothing new gets built until real users answer.
