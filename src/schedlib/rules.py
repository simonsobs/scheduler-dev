from typing import Tuple
import numpy as np
from chex import dataclass
from functools import partial
from abc import ABC
import datetime as dt

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
    (day, day_mod): (0, 1) means everyday, (4, 7) means every 4th day in a week, ...
    """
    day: int
    day_mod: int
    day_ref: dt.datetime
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda block: self.get_day_index(block.t0) % self.day_mod == self.day
        return core.seq_filter(filt, blocks)
    def get_day_index(self, t: dt.datetime) -> int:
        return np.floor((t - self.day_ref).total_seconds() / src.sidereal_day).astype(int)

@dataclass(frozen=True)
class DriftMode(Rule):
    """Restrict the blocks to a specific drift mode.
    
    Parameters
    ----------
    mode : str. drift mode ['rising', 'setting', 'both']
    """
    mode: str
    def __post_init__(self):
        if self.mode not in ['rising', 'setting', 'both']:
            raise ValueError(f"mode must be 'rising', 'setting' or 'both', got {self.mode}")
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda b: b if b.mode == self.mode else None
        return core.seq_map_when(core.block_isa(src.SourceBlock), filt, blocks)

@dataclass(frozen=True)
class MinDuration(Rule):
    """Restrict the minimum block size."""
    min_duration: int  # in seconds
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda block: block.duration >= self.min_duration
        return core.seq_filter(filt, blocks)

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

@dataclass(frozen=True)
class SunAvoidance(Rule):
    """Avoid sources that are too close to the Sun.

    Parameters
    ----------
    min_angle_az : float. minimum angle in deg
    min_angle_alt: float. minimum angle in deg
    n_buffer: int. number of time steps to buffer the sun mask
    time_step : int. time step in seconds
    """
    min_angle_az: float
    min_angle_alt: float
    n_buffer: int
    time_step: int

    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        sun_blocks = core.seq_map(src.block_get_matching_sun_block, blocks)
        return core.seq_map(self._apply_block, blocks, sun_blocks)

    def _apply_block(self, block: core.Block, sun_block: core.Block) -> core.Blocks:
        """Calculate the distance between a block and a sun block."""
        if isinstance(block, inst.ScanBlock):
            _, az, alt = sun_block.get_az_alt(time_step = dt.timedelta(seconds=self.time_step))
            az, alt = np.rad2deg(az), np.rad2deg(alt)
            daz, dalt = ((block.az - az) + 180) % 360 - 180, block.alt - alt
            daz, dalt = np.abs(daz) - block.throw, np.abs(dalt)
            ok = np.logical_or(daz > self.min_angle_az, dalt > self.min_angle_alt)
            safe_intervals = utils.ranges_complement(utils.ranges_pad(utils.mask2ranges(~ok), self.n_buffer, len(az)), len(az))
            # if the whole block is safe, return it
            if np.alltrue(safe_intervals == [[0, len(az)]]): return block
            # otherwise, split it up into safe intervals
            return [block.replace(t0=t0, t1=t1) for t0, t1 in safe_intervals]
        else:
            # passthrough
            return block

# global registry of rules
RULES = {
    'alt-range': AltRange,
    'day-mod': DayMod,
    'drift-mode': DriftMode,
    'min-duration': MinDuration,
    'rephase-first': RephaseFirst,
    'sun-avoidance': SunAvoidance,
}
def get_rule(name: str) -> Rule:
    return RULES[name]

def make_rule(name: str, **kwargs) -> Rule:
    return get_rule(name)(**kwargs)
