"""Eval runner.

The *harness* here is real (paired items, N independent runs). The *score
source* is a SIMULATION so the value test runs with zero API keys and zero cost.

Realistic noise model — this is what makes REAL-vs-NOISE discrimination
meaningful:
  observed_score(item) = config_mean(pipeline)          # depends on the change
                       + item_difficulty(item)          # SHARED across versions (paired)
                       + run_noise(item, run)            # INDEPENDENT per evaluation

Pairing cancels item_difficulty (the same question is asked of both versions) but
NOT run_noise (each generation is independently stochastic). So a tiny config
difference stays buried in noise and reads NOISE, while a genuine one clears the
band and reads REAL — exactly the behaviour a trustworthy tool needs.

>>> TO GO LIVE: replace the ``config_mean`` functions in ``SIGNALS`` with real
    evaluators (ragas, DeepEval, promptfoo, an LLM-judge). The harness, pairing
    and stats are already production-shaped. <<<
"""
from __future__ import annotations

import math

from ..core.schema import Pipeline

_EMB_Q = {
    "text-embedding-3-large": 0.90,
    "text-embedding-3-small": 0.80,
    "text-embedding-ada-002": 0.65,
}
_LLM_Q = {
    "gpt-4o": 0.85,
    "gpt-4o-mini": 0.75,
    "claude-sonnet-4": 0.86,
}

_DIFFICULTY_SD = 0.12   # spread across questions (shared -> cancelled by pairing)
_RUN_SD = 0.05          # per-run sampling noise (independent -> the real noise floor)


def _p(pipeline: Pipeline, node_type: str, key: str, default):
    for n in pipeline.nodes:
        if n.type == node_type:
            return n.params.get(key, default)
    return default


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fnv(*parts) -> int:
    """Stable, process-independent hash (FNV-1a) for seeding."""
    h = 2166136261
    for p in parts:
        for ch in str(p):
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def _gauss(seed: int, sd: float) -> float:
    import random
    return random.Random(seed).gauss(0.0, sd)


# The following two terms are ZERO at the default config (chunk size 512, cosine)
# so they leave the original example/tests unchanged, but let the value-test
# scenarios vary chunking and index metric realistically.
def _chunk_penalty(pipeline: Pipeline) -> float:
    """Quality loss from moving away from a ~512-token chunk sweet spot."""
    size = int(_p(pipeline, "chunker", "size", 512))
    return 0.10 * abs(math.log2(max(size, 1) / 512))


def _metric_penalty(pipeline: Pipeline) -> float:
    metric = _p(pipeline, "vectorstore", "metric", "cosine")
    return {"cosine": 0.0, "dot_product": 0.015, "euclidean": 0.03}.get(metric, 0.02)


# --- config_mean functions: expected score for a config (no per-item noise) ---
def context_recall(pipeline: Pipeline) -> float:
    emb = _EMB_Q.get(_p(pipeline, "embedder", "model", ""), 0.75)
    top_k = int(_p(pipeline, "retriever", "top_k", 4))
    return (0.50 + 0.28 * emb + 0.04 * math.log2(max(top_k, 1))
            - _chunk_penalty(pipeline) - _metric_penalty(pipeline))


def context_precision(pipeline: Pipeline) -> float:
    # Precision FALLS as top_k rises (more retrieved chunks dilute relevance) —
    # a deliberate tradeoff against recall, so a single change can help one metric
    # and hurt another.
    emb = _EMB_Q.get(_p(pipeline, "embedder", "model", ""), 0.75)
    top_k = int(_p(pipeline, "retriever", "top_k", 4))
    return (0.90 + 0.12 * emb - 0.05 * math.log2(max(top_k, 1))
            - 0.5 * _chunk_penalty(pipeline) - _metric_penalty(pipeline))


def faithfulness(pipeline: Pipeline) -> float:
    emb = _EMB_Q.get(_p(pipeline, "embedder", "model", ""), 0.75)
    llm = _LLM_Q.get(_p(pipeline, "llm", "model", ""), 0.78)
    # grounding depends on retrieved-context quality (emb) and generator (llm),
    # NOT on top_k -> supports honest multi-candidate attribution downstream.
    return 0.45 + 0.25 * llm + 0.30 * emb - _chunk_penalty(pipeline) - _metric_penalty(pipeline)


def answer_correctness(pipeline: Pipeline) -> float:
    emb = _EMB_Q.get(_p(pipeline, "embedder", "model", ""), 0.75)
    llm = _LLM_Q.get(_p(pipeline, "llm", "model", ""), 0.78)
    top_k = int(_p(pipeline, "retriever", "top_k", 4))
    return (0.40 + 0.25 * llm + 0.28 * emb + 0.02 * math.log2(max(top_k, 1))
            - _chunk_penalty(pipeline))


SIGNALS = {
    "context_recall": context_recall,
    "context_precision": context_precision,
    "faithfulness": faithfulness,
    "answer_correctness": answer_correctness,
}


def run_eval(
    eval_id: str, pipeline: Pipeline, n_items: int = 100, run_seed: int = 0
) -> list[float]:
    """Score ``pipeline`` on ``n_items`` items for one run.

    ``run_seed`` must differ between the two versions being compared so their
    sampling noise is independent; item difficulty is keyed only by item index so
    it is shared (paired) across versions.
    """
    mean_fn = SIGNALS[eval_id]
    mean = mean_fn(pipeline)
    scores = []
    for i in range(n_items):
        difficulty = _gauss(_fnv(eval_id, "item", i), _DIFFICULTY_SD)
        run_noise = _gauss(_fnv(eval_id, "run", run_seed, i), _RUN_SD)
        scores.append(_clamp(mean + difficulty + run_noise))
    return scores
