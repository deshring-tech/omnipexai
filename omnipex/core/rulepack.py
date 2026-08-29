"""Rule-pack interface — the plugin boundary that keeps the core generic.

A rule-pack encodes *domain knowledge* for one class of AI system: given a
change to a component parameter, what staleness propagates to which downstream
component types. This is the actual IP of the product. The core engine
(``omnipex.core.invalidation``) is a dumb interpreter of these rules.

Adding support for agents / voice / multimodal later means writing a new
rule-pack, NOT changing the core.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .errors import OmnipexError
from .schema import EffectType


@dataclass(frozen=True)
class Rule:
    """One declarative row of an invalidation matrix.

    "A change to ``(node_type, param)`` produces ``effect`` on every node whose
    type is ``target_type``." For RECOMPUTE/ BEHAVIOR effects the target must be
    downstream of the change; NO_EFFECT rules are assertions of independence and
    apply regardless of position (used to *prove* something is safe)."""

    target_type: str
    effect: EffectType
    reason: str


class RulePack(ABC):
    name: str = "base"

    @abstractmethod
    def rules_for(self, node_type: str, param: str) -> list[Rule]:
        """Return the invalidation rules triggered by changing ``param`` on a
        node of type ``node_type``. Empty list = change is not modelled."""

    def handles(self, node_type: str) -> bool:
        return True


_REGISTRY: dict[str, RulePack] = {}


def register(pack: RulePack) -> RulePack:
    _REGISTRY[pack.name] = pack
    return pack


def get(name: str) -> RulePack:
    if name not in _REGISTRY:
        raise OmnipexError(
            f"no rule-pack named {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
