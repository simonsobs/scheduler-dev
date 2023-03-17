from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class Rule: 
    pass

@dataclass(frozen=True)
class RuleSet:
    rules: FrozenSet[Rule]

@dataclass
class Policy:
    pass
