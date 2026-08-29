"""Generic, domain-agnostic representation of an AI pipeline.

Nothing in this module knows what an "embedder" or a "retriever" is. It only
knows about typed nodes, directed data-flow edges, and the three kinds of
staleness a change can propagate. ALL domain knowledge lives in rule-packs
(see ``omnipex.rulepacks``) so that new AI system types (agents, voice,
multimodal, ...) are added as *data*, not as rewrites of this core.

This is the "design for generality, build for specificity" boundary: the schema
is generic; today only the RAG rule-pack is implemented against it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import PipelineValidationError, UnsupportedChangeError


class EffectType(str, Enum):
    """The three kinds of staleness a change can propagate downstream."""

    NO_EFFECT = "no_effect"                    # provably independent
    RECOMPUTE_REQUIRED = "recompute_required"  # a stored artifact is now physically stale
    BEHAVIOR_MAY_CHANGE = "behavior_may_change"  # output quality can shift; re-evaluate


@dataclass
class Node:
    """A typed component in the pipeline. ``type`` is an opaque string to the
    core; rule-packs give it meaning."""

    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    produces_artifact: str | None = None   # e.g. "vector_index"
    evals: list[str] = field(default_factory=list)  # eval ids bound to this node
    # Path (relative to the manifest) of a build-emitted stamp recording what the
    # artifact was ACTUALLY built with. Enables consistency checking; see
    # ``core.consistency``.
    stamp: str | None = None
    # Environment variables this component needs at run time.
    requires_env: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    name: str
    nodes: list[Node]
    edges: list[tuple[str, str]]  # (source_id, target_id); data flows source -> target

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise PipelineValidationError(
            f"pipeline {self.name!r} has no node with id {node_id!r}"
        )

    def validate(self) -> None:
        """Reject structurally invalid pipelines with a precise message.

        Checks: unique node ids, edges reference existing nodes, graph is a DAG.
        Called by the manifest loader; call it yourself when constructing
        pipelines programmatically.
        """
        seen: set[str] = set()
        for n in self.nodes:
            if n.id in seen:
                raise PipelineValidationError(
                    f"pipeline {self.name!r}: duplicate node id {n.id!r}"
                )
            seen.add(n.id)

        indegree = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for src, dst in self.edges:
            for end in (src, dst):
                if end not in seen:
                    raise PipelineValidationError(
                        f"pipeline {self.name!r}: edge ({src!r} -> {dst!r}) "
                        f"references unknown node {end!r}"
                    )
            adj[src].append(dst)
            indegree[dst] += 1

        # Kahn's algorithm: if not every node can be topologically ordered,
        # the graph has a cycle and staleness propagation is ill-defined.
        queue = deque(nid for nid, deg in indegree.items() if deg == 0)
        ordered = 0
        while queue:
            cur = queue.popleft()
            ordered += 1
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if ordered != len(self.nodes):
            cyclic = sorted(nid for nid, deg in indegree.items() if deg > 0)
            raise PipelineValidationError(
                f"pipeline {self.name!r}: dependency cycle involving {cyclic}"
            )

    def reachable_from(self, node_id: str) -> set[str]:
        """All nodes downstream of (and including) ``node_id`` via forward edges."""
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, []))
        return seen


@dataclass
class Change:
    node_id: str
    node_type: str
    param: str
    old: Any
    new: Any

    def __str__(self) -> str:
        return f"{self.node_id}.{self.param}: {self.old!r} -> {self.new!r}"


def diff_pipelines(base: Pipeline, cand: Pipeline) -> list[Change]:
    """Detect per-parameter changes between two versions of a pipeline.

    Matches nodes by id. Structural changes (nodes added/removed, or a node
    changing type) are not analyzable in this version; they raise
    ``UnsupportedChangeError`` rather than being silently ignored, so a result
    is never quietly computed from a partial diff.
    """
    base_ids = {n.id for n in base.nodes}
    cand_ids = {n.id for n in cand.nodes}
    if base_ids != cand_ids:
        raise UnsupportedChangeError(
            "structural changes are not supported yet: "
            f"added={sorted(cand_ids - base_ids)}, "
            f"removed={sorted(base_ids - cand_ids)}. "
            "This version analyzes parameter changes between structurally "
            "identical pipelines."
        )

    changes: list[Change] = []
    base_by_id = {n.id: n for n in base.nodes}
    for cand_node in cand.nodes:
        base_node = base_by_id[cand_node.id]
        if base_node.type != cand_node.type:
            raise UnsupportedChangeError(
                f"node {cand_node.id!r} changed type "
                f"({base_node.type!r} -> {cand_node.type!r}); type changes are "
                "structural and not supported yet."
            )
        keys = set(base_node.params) | set(cand_node.params)
        for key in sorted(keys):
            old = base_node.params.get(key)
            new = cand_node.params.get(key)
            if old != new:
                changes.append(
                    Change(cand_node.id, cand_node.type, key, old, new)
                )
    return changes
