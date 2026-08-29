"""Difficulty-graded scenarios for the value test.

Each scenario is one commit that changed several pipeline params; a metric then
looks worse. The participant must find the culprit change (or conclude it is
noise). Scenarios span a difficulty gradient on purpose — the point is to locate
*where* Omnipex starts to matter:

  easy    a single change; intuition alone probably suffices (Omnipex adds little)
  medium  two changes, one innocent; intuition is shaky
  hard    the intuitive suspect is wrong, or a real drop is masked by an
          improvement, or the "regression" is actually noise (a wild-goose chase)

Ground truth is authored explicitly AND cross-checked against the attribution
engine (validate_all), so a bad scenario fails loudly instead of silently.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import omnipex.rulepacks  # noqa: F401  (registers the built-in rule-packs)
from omnipex.core import attribution
from omnipex.core.rulepack import get
from omnipex.core.schema import Change, Node, Pipeline, diff_pipelines

RAG = get("rag")


def base_pipeline() -> Pipeline:
    """A RAG pipeline with all four quality metrics bound (independent of the
    shipped example so experiment changes never disturb product tests)."""
    return Pipeline(
        name="docs_qa_exp",
        nodes=[
            Node("chunker", "chunker", {"size": 512, "overlap": 50}),
            Node("embedder", "embedder",
                 {"model": "text-embedding-3-small", "dim": 1536},
                 produces_artifact="vector_index"),
            Node("store", "vectorstore", {"metric": "cosine"}),
            Node("retriever", "retriever", {"top_k": 4},
                 evals=["context_recall", "context_precision"]),
            Node("prompt", "prompt", {"template_hash": "abc123"}),
            Node("llm", "llm", {"model": "gpt-4o", "temperature": 0.0},
                 evals=["faithfulness", "answer_correctness"]),
        ],
        edges=[("chunker", "embedder"), ("embedder", "store"),
               ("store", "retriever"), ("retriever", "prompt"), ("prompt", "llm")],
    )


@dataclass
class Scenario:
    id: str
    title: str
    difficulty: str            # easy | medium | hard
    metric: str                # the metric under investigation
    changes: list[tuple[str, str, Any]]   # (node_id, param, new_value)
    expected: str | None       # ground-truth culprit "node_id.param", or None = noise
    note: str = ""             # what makes it interesting (for the facilitator only)

    def candidate(self) -> Pipeline:
        p = copy.deepcopy(base_pipeline())
        for node_id, param, new in self.changes:
            p.node(node_id).params[param] = new
        return p

    def change_objects(self) -> list[Change]:
        return diff_pipelines(base_pipeline(), self.candidate())


def change_key(c: Change) -> str:
    return f"{c.node_id}.{c.param}"


SCENARIOS: list[Scenario] = [
    Scenario(
        id="s1_single_embedder", title="Embedding model swap", difficulty="easy",
        metric="faithfulness",
        changes=[("embedder", "model", "text-embedding-ada-002")],
        expected="embedder.model",
        note="Single change; a strong engineer should get this from intuition. "
             "Baseline case — expect little Omnipex advantage.",
    ),
    Scenario(
        id="s2_embedder_plus_topk", title="Retrieval tuning + embedder swap",
        difficulty="medium", metric="faithfulness",
        changes=[("embedder", "model", "text-embedding-ada-002"),
                 ("retriever", "top_k", 8)],
        expected="embedder.model",
        note="Two changes; top_k is innocent for faithfulness but looks like a "
             "plausible suspect. Manual arm must rule it out.",
    ),
    Scenario(
        id="s3_masked_correctness", title="The deceptive upgrade",
        difficulty="hard", metric="answer_correctness",
        changes=[("embedder", "model", "text-embedding-ada-002"),
                 ("retriever", "top_k", 8)],
        expected="embedder.model",
        note="top_k bump IMPROVES correctness and partially masks the embedder "
             "regression, so the net drop looks small/ambiguous. High-value case.",
    ),
    Scenario(
        id="s4_chunk_vs_model", title="Chunking change + model downgrade",
        difficulty="hard", metric="faithfulness",
        changes=[("chunker", "size", 1024),
                 ("llm", "model", "gpt-4o-mini")],
        expected="chunker.size",
        note="Intuition blames the model swap; the bigger real driver is the "
             "chunk-size change. Both are real but chunker dominates.",
    ),
    Scenario(
        id="s5_precision_tradeoff", title="Recall/precision tradeoff",
        difficulty="medium", metric="context_precision",
        changes=[("retriever", "top_k", 12)],
        expected="retriever.top_k",
        note="Single change; recall went UP while precision went DOWN. Tests "
             "whether the manual arm trusts a counterintuitive direction.",
    ),
    Scenario(
        id="s6_noise_trap", title="Is it even real?", difficulty="hard",
        metric="faithfulness",
        changes=[("llm", "temperature", 0.1), ("retriever", "top_k", 5)],
        expected=None,
        note="Neither change affects faithfulness; the dip is pure noise. Correct "
             "answer is 'nothing — noise'. Tests false-attribution / wild-goose chase.",
    ),
]


def ground_truth(scenario: Scenario) -> str | None:
    """What the simulated world actually says (the answer key)."""
    report = attribution.attribute(base_pipeline(), scenario.candidate(), RAG)
    for ma in report.metrics:
        if ma.metric == scenario.metric and ma.culprit is not None:
            return f"{ma.culprit.node_id}.{ma.culprit.param}"
    return None


def validate_all() -> list[str]:
    """Return a list of mismatch messages (empty = all scenarios consistent)."""
    problems = []
    for s in SCENARIOS:
        truth = ground_truth(s)
        if truth != s.expected:
            problems.append(
                f"{s.id}: authored expected={s.expected!r} but engine says {truth!r}"
            )
    return problems


def get_scenario(scenario_id: str) -> Scenario:
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(scenario_id)
