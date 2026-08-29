# OmniPex — Vision & Roadmap

The single canonical strategy document. It holds both halves honestly: the
**destination** (an all-in-one workspace for AI engineers) and the **route**
(enter through the one capability nobody owns, and earn the rest).

**Operating rule:** stages are entered in order, and each is gated on the one
before it. Nothing in "Deferred" gets built until the engine has users.

---

## 1. The destination

AI engineering is fragmented. Engineers switch between editors, eval frameworks,
experiment trackers, dataset tools, deployment platforms, and dashboards all day,
and every switch costs context. The long-term goal is one workspace where
building, evaluating, debugging, and shipping AI systems happen in one place,
with AI as a working partner throughout.

That destination is real and it stays. What follows is the argument about *where
to enter it* — because entry point, not ambition, is what decides whether a
platform ever gets built.

## 2. Why we enter at the intelligence layer

The front of the AI-engineering workflow — editor, project explorer, code
assistance — is the most contested software market on earth: VS Code, Cursor
(~$29B), GitHub Copilot, Cognition/Windsurf (~$26B). Entering there means
competing on their turf, with their funding, for a user who has no reason to
switch.

The **end** of the workflow is unowned. Nobody models *behavioural invalidation*:
change the embedder, and retrieval quality silently drifts three steps
downstream. Eval frameworks (DeepEval, RAGAS, LangSmith, Braintrust, Phoenix)
measure outputs; code tools (Codemaps) map call graphs. Neither knows which
config change made which downstream metric stale, nor which of several
simultaneous changes caused a regression.

> **Positioning:** LangSmith tells you your eval scores. OmniPex tells you *which
> change moved them* — before you ship.

Workflow order is not roadmap order. You build where the pain is unowned, not
where the user's day begins. AWS was rent-a-server; Stripe was a payments API;
Databricks was hosted Spark. Every platform was one narrow indispensable thing
first.

**Category check:** OmniPex is not Photoshop (a creation tool). It is Sentry, or
Datadog, or git bisect — an intelligence layer over work that happens elsewhere.
Judged as an IDE it looks empty; judged as regression intelligence it is complete.

## 3. The moat

The hand-authored invalidation matrix (`omnipex/rulepacks/rag.py`) is v1 — senior
engineering knowledge encoded as data. The **moat** is correcting it *empirically*
from observed eval deltas across many real pipelines. That requires accumulated
observation of real systems changing over time, which cannot be cloned from a
screenshot. Continuity is the asset — which is also why the paid tier is the
history layer (§6).

## 4. Roadmap

### Stage 1 — Intelligence layer ✅ built
Invalidation graph, change-impact analysis (`diff`), bisection attribution
(`blame`), consistency checking (`doctor`), paired-bootstrap noise bands, CI exit
codes, local GUI, MCP server, service layer (`api.py`) shared by every interface.
Production-hardened: validated manifests, error hierarchy, family-wise false
discovery control, 34 tests.
*Eval scores are still simulated; the swap point is `SIGNALS` in
`omnipex/eval/runner.py`.*

### Stage 2 — Validation ⬅️ **current gate**
Timed control-vs-treatment experiment with 8–12 working AI engineers
(`valuetest/`, see `PROTOCOL.md` and `RECRUITING.md`). Measures time-to-culprit,
accuracy, and confidence against today's manual revert-and-rerun workflow.
- **Kill-criterion:** if OmniPex does not measurably reduce time-to-culprit *and*
  engineers do not report the manual workflow as painful, the thesis is wrong.
  Learned in weeks, not a year.
- Also pending: derive the CI penalty empirically (`timing/time_eval.py`), and
  blind `valuetest/scenarios.py` before any self-run kit leaves the machine.

### Stage 3 — Make it real *(gated on Stage 2 PASS)*
1. **Live eval backend** — replace simulated scores with DeepEval. Validates
   attribution correctness on a real pipeline (a separate hypothesis from the
   workflow one).
2. **Vendor adapters for `doctor`** — the stamp-file mechanism shipped early in
   Stage 1; adapters that read live index metadata (Pinecone, Chroma, Qdrant,
   Weaviate) come here, as optional extras with lazy imports so the core stays
   dependency-free.
3. **Auto-extraction** — rebuild the graph from OpenInference/OpenTelemetry
   traces instead of a hand-written manifest. This is the "point it at my project
   and the pipeline appears, already connected" moment.
   **Kill-criterion:** if a correct graph cannot be auto-built for 3 of 5 real
   open-source RAG repos without per-repo hacking, retreat to the manifest
   product — smaller, still viable. Do not chase zero-config forever.

### Stage 4 — Distribution *(partially built)*
Meet engineers inside the tools they already run — layer on top, never ask anyone
to switch.
- ✅ **MCP server** (`omnipex mcp`) — Claude Code / Cursor call the engine
  directly. Stdlib JSON-RPC over stdio; four tools mapping onto the CLI.
- Remaining: **GitHub App** posting the blame comment on the PR, **VS Code
  extension**, **CI action**.

### Stage 5 — Flywheel *(gated on Stage 4)*
Log real changes and observed deltas; compare against the hand-authored matrix.
**Signal wanted:** the data *disagrees* with a hand-written row — that
disagreement is proprietary knowledge accumulating.

### Stage 6 — Workspace shell *(earned, not assumed)*
Projects, pipeline views, health, history, collaboration — built *around* a
capability engineers already depend on. At this point it is not "another IDE
asking you to switch," it is the natural interface to something already in use.
Docker got Docker Desktop; AWS got a console. Engine first, shell second.

### Stage 7 — The platform
Adjacent modules as separate rule-packs and views: agents, voice, multimodal,
datasets, experiments, deployment. The all-in-one workspace, reached floor by
floor.

## 5. Architecture principle that keeps this possible

**Generic core, pluggable domain packs; one engine, many views.**
`omnipex/core/` knows nothing about RAG — a new AI-system class is a new
rule-pack, never a rewrite. `omnipex/api.py` is the single source of truth; the
CLI, GUI, and any future MCP server or HTTP API are *views* that may not compute
results themselves. A GUI-only capability would break this and must not exist.

## 6. Business model

Developer-tool PLG: self-serve, freemium, no sales motion.

| Free (local, unlimited) | Paid (self-serve) |
|---|---|
| CLI, GUI, `diff`, `blame`, `doctor` | Team history across commits |
| Single dev, single pipeline | PR comments in CI |
| Full engine, runs offline | Shared/learned invalidation matrices |
| | Org-wide flywheel & dashboards |

The paid tier is **continuity** — precisely what a local one-shot tool cannot
provide, and the same thing that is the moat.

Notes: a hosted upload-a-file service was considered and rejected — eval
execution and bisection require sitting next to the pipeline, and teams will not
send configs or quality scores to a third party. Local-first is a feature. A
public **playground** (prebuilt scenarios, no upload, instant "aha") is the
right frictionless front door and doubles as a recruiting asset. License is
undeclared pending an open-core decision; open-sourcing publishes the seed
matrix but not the learned corrections.

## 7. Deferred — good ideas, wrong time

Revisit once the engine has users. Not rejected; sequenced.

- **Generalised invalidation ontology** (Component → Capability → Behaviour →
  Business metric). A general abstraction must be *derived* from 2–3 concrete
  rule-packs, never invented up front. Core schema is already generic; add packs
  one vertical at a time.
- **"Project Brain"** — nodes storing purpose, owner, decisions, discussions.
  Different and weaker trust model: human-maintained prose rots, and one stale
  field costs trust in the eval numbers that were correct. A later floor.
- **Predictive / business-KPI intelligence** — "which KPI moves if I merge this?"
  The north star, and possibly never fully solvable: production attribution is
  the offline noise problem times confounding. Ambition, not critical path.
- **Datasets, training, fine-tuning, deployment, monitoring, agent builder** —
  each a real module in the destination; each gated behind a validated engine.

## 8. Deliberately not building (recommendation, revisitable)

- **Code editor / AI coding assistant.** Head-on with Cursor, Copilot, and
  Windsurf — unwinnable for a solo founder, and it converts a defensible
  intelligence layer into an undifferentiated IDE. The workspace (Stage 6) can
  exist without an editor inside it.
- **AI education / interactive learning / mentor.** A different customer
  (beginners) with different GTM and product DNA. Our user is an engineer who has
  already shipped a pipeline. Serving both sharply serves neither.

## 9. The only metric that matters right now

Would anyone be *angry* if this disappeared? Build nothing new until real users
answer yes.
