"""The invalidation engine: propagate a change through the graph using a
rule-pack. Domain-agnostic — all knowledge comes from the rule-pack.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rulepack import RulePack
from .schema import Change, EffectType, Pipeline, diff_pipelines


@dataclass
class Effect:
    node_id: str
    node_type: str
    effect: EffectType
    reason: str
    cause: Change


def propagate(pipeline: Pipeline, change: Change, pack: RulePack) -> list[Effect]:
    """Compute the effects of a single change."""
    reachable = pipeline.reachable_from(change.node_id)
    effects: list[Effect] = []
    for rule in pack.rules_for(change.node_type, change.param):
        for node in pipeline.nodes:
            if node.type != rule.target_type:
                continue
            # RECOMPUTE / BEHAVIOR can only reach nodes downstream of the change.
            # NO_EFFECT is an explicit safety assertion (often about upstream
            # artifacts) so it is reported regardless of reachability.
            if rule.effect is not EffectType.NO_EFFECT and node.id not in reachable:
                continue
            effects.append(
                Effect(node.id, node.type, rule.effect, rule.reason, change)
            )
    return effects


@dataclass
class Report:
    changes: list[Change]
    effects: list[Effect]
    # node_id -> artifact name that must be rebuilt
    recompute: dict[str, str] = field(default_factory=dict)
    # eval_id -> (node_id, [candidate cause changes])
    stale_evals: dict[str, tuple[str, list[Change]]] = field(default_factory=dict)
    confirmed_safe: list[Effect] = field(default_factory=list)


def analyze(base: Pipeline, cand: Pipeline, pack: RulePack) -> Report:
    """Full change-impact analysis between two pipeline versions."""
    changes = diff_pipelines(base, cand)
    all_effects: list[Effect] = []
    for change in changes:
        all_effects.extend(propagate(cand, change, pack))

    report = Report(changes=changes, effects=all_effects)

    # Nodes with any real effect can never also be "proven safe".
    affected = {
        eff.node_id
        for eff in all_effects
        if eff.effect is not EffectType.NO_EFFECT
    }
    for eff in all_effects:
        node = cand.node(eff.node_id)
        if eff.effect is EffectType.RECOMPUTE_REQUIRED:
            report.recompute[eff.node_id] = node.produces_artifact or node.type
        elif eff.effect is EffectType.NO_EFFECT and eff.node_id not in affected:
            report.confirmed_safe.append(eff)

    # A node's evals go stale iff its behaviour may change. Attribution = every
    # change whose blast radius reached this node (honest: may be >1 candidate).
    for eff in all_effects:
        if eff.effect is not EffectType.BEHAVIOR_MAY_CHANGE:
            continue
        node = cand.node(eff.node_id)
        for eval_id in node.evals:
            _, causes = report.stale_evals.get(eval_id, (eff.node_id, []))
            if eff.cause not in causes:
                causes.append(eff.cause)
            report.stale_evals[eval_id] = (eff.node_id, causes)

    return report
