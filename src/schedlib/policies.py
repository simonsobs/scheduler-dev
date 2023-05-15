#!/usr/bin/env python3

from chex import dataclass
from typing import Dict
import datetime as dt
from . import core, utils

@dataclass(frozen=True)
class BasicPolicy(core.Policy):
    master_schedule: str
    # block_tolerance: Dict[str, float]
    # def get_block_tolerance(self, block:core.Block) -> float:
    #     # avoid silent default for clarity
    #     if str(type(block)) not in self.block_tolerance:
    #         raise ValueError(f"Block type {str(type(block))} not in block_tolerance but a tolerance is asked")
    #     return self.block_tolerance[str(type(block))]

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        blocks = utils.parse_sequence_from_toast(self.master_schedule)
        return core.seq_trim(blocks, t0, t1)

    def apply(self, blocks: core.BlocksTree) -> core.Blocks:
        # dummy policy: just return the blocks
        return core.seq_flatten(blocks)