from typing import Tuple
import numpy as np
from chex import dataclass
from functools import partial
from abc import ABC

from . import core, source as src, instrument as inst, utils

@dataclass(frozen=True)
class Rule(core.BlocksTransformation, ABC):
    """Guarantee that our rule preserves nested structure."""
    def __call__(self, blocks: core.BlocksTree) -> core.BlocksTree:
        out = self.apply(blocks)
        assert core.seq_is_nested(out) == core.seq_is_nested(blocks), "Rule must preserve nested structure"
        return out

@dataclass(frozen=True)
class AltRange(Rule):
    """Restrict the altitude range of source blocks. 

    Parameters
    ----------
    alt_range : Tuple[float, float]. min and max altitude in degrees 
    """
    alt_range: Tuple[float, float]
    def apply(self, blocks:core.BlocksTree) -> core.BlocksTree:
        filt = partial(src.source_block_trim_by_az_alt_range, alt_range=np.deg2rad(self.alt_range))
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt, blocks)

@dataclass(frozen=True)
class DayMod(Rule):
    """Restrict the blocks to a specific day of the week.
    
    Parameters
    ----------
    day : int. 0 is Monday, 6 is Sunday
    """
    day: int
    def __post_init__(self):
        if self.day not in range(7):
            raise ValueError(f"day must be in range(7), got {self.day}") 
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda b: b.t0.weekday() == self.day
        return core.seq_filter(filt, blocks)

@dataclass(frozen=True)
class DriftMode(Rule):
    """Restrict the blocks to a specific drift mode.
    
    Parameters
    ----------
    mode : str. drift mode ['rising', 'setting']
    """
    mode: str
    def __post_init__(self):
        if self.mode not in ['rising', 'setting']:
            raise ValueError(f"mode must be 'rising' or 'setting', got {self.mode}") 
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda b: b if b.mode == self.mode else None
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt, blocks)

@dataclass(frozen=True)
class RephaseFirst(Rule):
    """Randomize the phase of the first block"""
    max_fraction: float
    min_block_size: float  # in seconds
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        if len(blocks) == 0: return blocks
        # identify the first block as the first in the sorted list
        src = core.seq_sort(core.seq_flatten(blocks))[0]
        # randomize the phase of it but not too much
        allowance = min(self.max_fraction * src.duration,
                        src.duration - self.min_block_size,
                        0)
        tgt = src.replace(t0=src.t0 + np.random.uniform(0, allowance))
        return core.seq_replace_block(blocks, src, tgt)

# global registry of rules
RULES = {
    'alt-range': AltRange,
    'day-mod': DayMod,
    'drift-mode': DriftMode,
    'rephase-first': RephaseFirst,
}
def get_rule(name: str) -> Rule:
    return RULES[name]

def make_rule(name: str, **kwargs) -> Rule:
    return get_rule(name)(**kwargs)
