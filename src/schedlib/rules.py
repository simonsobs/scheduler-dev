from typing import Tuple
import numpy as np
from chex import dataclass
from functools import partial

from . import core, source as src

Rule = core.BlocksTransformation  # alias for readability

@dataclass(frozen=True)
class AltRange(Rule):
    """Restrict the altitude range of source blocks. 

    Parameters
    ----------
    alt_range : Tuple[float, float]. min and max altitude in degrees 
    """
    alt_range: Tuple[float, float]
    def apply(self, blocks:core.Blocks) -> core.Blocks:
        filt_alt = partial(src.source_block_trim_by_az_alt_range, alt_range=np.deg2rad(self.alt_range))
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt_alt, blocks)

@dataclass(frozen=True)
class AzRange(Rule):
    """Restrict the altitude range of source blocks. 

    Parameters
    ----------
    az_range : Tuple[float, float]. min and max altitude in degrees 
    """
    alt_range: Tuple[float, float]
    def apply(self, blocks:core.Blocks) -> core.Blocks:
        filt_alt = partial(src.source_block_trim_by_az_alt_range, az_range=np.deg2rad(self.az_range))
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt_alt, blocks)

@dataclass(frozen=True)
class WeekDay(Rule):
    """Restrict the blocks to a specific day of the week.
    
    Parameters
    ----------
    day : int. 0 is Monday, 6 is Sunday
    """
    day: int
    def __post_init__(self):
        if self.day not in range(7):
            raise ValueError(f"day must be in range(7), got {self.day}") 
    def apply(self, blocks: core.Blocks) -> core.Blocks:
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
    def apply(self, blocks: core.Blocks) -> core.Blocks:
        filt = lambda b: b if b.mode == self.mode else None
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt, blocks)

# global registry of rules
RULES = {
    'alt-range': AltRange,
    'az-range': AzRange,
    'weekday': WeekDay,
    'drift-mode': DriftMode,
}
def get_rule(name: str) -> Rule:
    return RULES[name]

def make_rule(name: str, **kwargs) -> Rule:
    return get_rule(name)(**kwargs)