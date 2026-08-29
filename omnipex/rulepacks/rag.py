"""RAG rule-pack: the invalidation matrix for retrieval-augmented generation.

THIS TABLE IS THE PRODUCT. Every row is a piece of senior-AI-engineer knowledge
encoded as data: "if you change X, here is exactly what goes stale downstream and
what stays safe." Today it is hand-authored. The long-term moat is learning /
correcting these rows empirically from observed eval deltas across real
pipelines (the data flywheel) — but hand-authored is enough to test whether the
core loop delivers value.

Component types this pack understands:
    chunker -> embedder -> vectorstore -> retriever -> prompt -> llm
"""
from __future__ import annotations

from ..core.rulepack import Rule, RulePack, register
from ..core.schema import EffectType as E

# (changed_node_type, param) -> list[Rule]
RAG_MATRIX: dict[tuple[str, str], list[Rule]] = {
    ("embedder", "model"): [
        Rule("embedder", E.RECOMPUTE_REQUIRED,
             "new embedding model -> the entire corpus must be re-embedded"),
        Rule("vectorstore", E.RECOMPUTE_REQUIRED,
             "stored vectors were produced by the old model; the index is invalid"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE,
             "nearest-neighbour ranking shifts in a new embedding space"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE,
             "different context is retrieved, so answers may change"),
    ],
    ("embedder", "dim"): [
        Rule("embedder", E.RECOMPUTE_REQUIRED, "embedding dimensionality changed"),
        Rule("vectorstore", E.RECOMPUTE_REQUIRED,
             "index dimensionality must match; rebuild required"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "ranking changes"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "retrieved context changes"),
    ],
    ("chunker", "size"): [
        Rule("chunker", E.RECOMPUTE_REQUIRED, "re-chunking moves every chunk boundary"),
        Rule("embedder", E.RECOMPUTE_REQUIRED, "new chunks must be re-embedded"),
        Rule("vectorstore", E.RECOMPUTE_REQUIRED, "index must be rebuilt from new chunks"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "retrieved spans change"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "answer context changes"),
    ],
    ("chunker", "overlap"): [
        Rule("chunker", E.RECOMPUTE_REQUIRED, "overlap change re-chunks the corpus"),
        Rule("embedder", E.RECOMPUTE_REQUIRED, "new chunks must be re-embedded"),
        Rule("vectorstore", E.RECOMPUTE_REQUIRED, "index must be rebuilt"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "retrieved spans change"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "answer context changes"),
    ],
    ("retriever", "top_k"): [
        Rule("vectorstore", E.NO_EFFECT,
             "index is unchanged; no re-embedding or reindexing needed"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "number of retrieved chunks changes"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "the model sees more/less context"),
    ],
    ("retriever", "reranker"): [
        Rule("vectorstore", E.NO_EFFECT, "index is unchanged"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "result ordering changes"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "answer context ordering changes"),
    ],
    ("prompt", "template_hash"): [
        Rule("retriever", E.NO_EFFECT,
             "retrieval is upstream of the prompt and is unaffected"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "the instructions given to the model changed"),
    ],
    ("llm", "model"): [
        Rule("retriever", E.NO_EFFECT, "retrieval does not depend on the generator"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "the generation model changed"),
    ],
    ("llm", "temperature"): [
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "sampling temperature changed"),
    ],
    ("vectorstore", "metric"): [
        Rule("vectorstore", E.RECOMPUTE_REQUIRED,
             "a different distance metric may require reindexing"),
        Rule("retriever", E.BEHAVIOR_MAY_CHANGE, "ranking uses a different distance"),
        Rule("llm", E.BEHAVIOR_MAY_CHANGE, "retrieved context may change"),
    ],
}


class RAGRulePack(RulePack):
    name = "rag"

    def rules_for(self, node_type: str, param: str) -> list[Rule]:
        return RAG_MATRIX.get((node_type, param), [])

    def handles(self, node_type: str) -> bool:
        return node_type in {
            "chunker", "embedder", "vectorstore", "retriever", "prompt", "llm"
        }


register(RAGRulePack())
