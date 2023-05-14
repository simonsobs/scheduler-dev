#!/usr/bin/env python3

from chex import dataclass
from . import core

@dataclass(frozen=True)
class BasicPolicy(core.Policy):
    def get_block_tolarance(self, block:core.Block) -> float:
        return self.get('block_tolarance', {}).get(str(type(block)), 0)
