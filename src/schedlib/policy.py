from typing import FrozenSet
from chex import dataclass


@dataclass(frozen=True)
class Rule: 
    pass

@dataclass(frozen=True)
class RuleSet:
    rules: FrozenSet[Rule]

@dataclass(frozen=True)
class Policy:
    config: dict

    @classmethod
    def from_config(cls, config: dict):
        return cls(config=config)

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

@dataclass(frozen=True)
class BasicPolicy(Policy):
    def get_block_tolarance(self, block):
        return self.get('block_tolarance', {}).get(str(type(block)), 0)
