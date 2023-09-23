from typing import Tuple, Dict, List, Optional
import numpy as np
from abc import ABC, abstractmethod
import datetime as dt
from dataclasses import dataclass

from . import core, source as src, instrument as inst, utils

@dataclass(frozen=True)
class GreenRule(core.BlocksTransformation, ABC):
    """GreenRule preserves trees"""
    def __call__(self, blocks: core.BlocksTree) -> core.BlocksTree:
        out = self.apply(blocks)
        assert core.seq_is_nested(out) == core.seq_is_nested(blocks), "GreenRule must preserve trees"
        return out

@dataclass(frozen=True)
class MappableRule(GreenRule, ABC):
    """MappableRule preserves sequences"""
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        return core.seq_map_when(self.applicable, self.apply_block, blocks)
    @abstractmethod
    def apply_block(self, block) -> core.Blocks: ...
    def applicable(self, block) -> bool: 
        return True

@dataclass(frozen=True)
class ConstrainedRule(GreenRule):
    """ConstrainedRule applies a rule to a subset of blocks. Here
    constraint is a fnmatch pattern that matches to the `key` of a
    block."""
    rule: core.Rule
    constraint: str
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        matched, unmatched = core.seq_partition_wth_query(self.constraint, blocks)
        return core.seq_combine(self.rule(matched), unmatched)

@dataclass(frozen=True)
class AltRange(MappableRule):
    """Restrict the altitude range of source blocks. 

    Parameters
    ----------
    alt_range : Tuple[float, float]. min and max altitude in degrees 
    """
    alt_range: Tuple[float, float]

    def apply_block(self, block:core.Block) -> core.Block:
        return block.trim_by_az_alt_range(alt_range=self.alt_range)

    def applicable(self, block:core.Block) -> bool:
        return isinstance(block, src.SourceBlock)

@dataclass(frozen=True)
class DayMod(GreenRule):
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
        return np.floor((t - self.day_ref).total_seconds() / utils.sidereal_day).astype(int)

@dataclass(frozen=True)
class DriftMode(MappableRule):
    """Restrict the blocks to a specific drift mode.
    
    Parameters
    ----------
    mode : str. drift mode ['rising', 'setting', 'both']
    """
    mode: str
    def __post_init__(self):
        if self.mode not in ['rising', 'setting', 'both']:
            raise ValueError(f"mode must be 'rising', 'setting' or 'both', got {self.mode}")
    def apply_block(self, block: core.Block) -> core.Block:
        return block if block.mode == self.mode else None
    def applicable(self, block: core.Block) -> bool:
        return isinstance(block, src.SourceBlock)

@dataclass(frozen=True)
class MinDuration(GreenRule):
    """Restrict the minimum block size."""
    min_duration: int  # in seconds
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        filt = lambda block: block.duration >= dt.timedelta(seconds=self.min_duration)
        return core.seq_filter(filt, blocks)

@dataclass(frozen=True)
class RephaseFirst(GreenRule):
    """Randomize the phase of the first block"""
    max_fraction: float
    min_block_size: float  # in seconds
    rng_key: utils.PRNGKey
    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        if len(blocks) == 0: return blocks
        # identify the first block as the first in the sorted list
        src = core.seq_sort(core.seq_flatten(blocks))[0]
        # randomize the phase of it but not too much
        allowance = min(self.max_fraction * src.duration.total_seconds(),
                        max(src.duration.total_seconds() - self.min_block_size, 0))
        tgt = src.replace(t0=src.t0 + dt.timedelta(seconds=utils.uniform(self.rng_key, 0, allowance)))
        return core.seq_replace_block(blocks, src, tgt)

@dataclass(frozen=True)
class MakeSourcePlan(MappableRule):
    """Convert source blocks to scan blocks"""
    specs: List[Dict[str, List[float]]]
    spec_shape: str
    max_obs_length: float
    bounds_alt: Optional[Tuple[float, float]] = None
    bounds_az_throw: Optional[Tuple[float, float]] = None

    def apply_block(self, block: core.Block):
        if block.mode == "both": return block  # not relevant
        # get some shape parameters
        pos_hi = max([spec['bounds_y'][1] for spec in self.specs])
        pos_lo = min([spec['bounds_y'][0] for spec in self.specs])
        alt_offset = (pos_hi + pos_lo) / 2  # offset from boresight
        alt_height = pos_hi - pos_lo

        # we can scan any time in the observation window
        # suppose we start at time t, the source will be at:
        t, az, alt = block.get_az_alt()

        # require at least two samples to interpolate:
        if len(t) < 2: return None  # filtered

        sign = {"rising": 1, "setting": -1}[block.mode]
        # the center of wafer set should be at this alt
        alt_center = alt + sign*alt_height/2
        # we should stop scanning when the source is at this alt
        alt_stop = alt + sign*alt_height
        # total passage time
        obs_length = utils.interp_extra(alt_stop, alt, t, fill_value=np.nan) - t
        ok = np.logical_not(np.isnan(obs_length))
        assert np.all(obs_length[ok] >= 0), "passage time must be positive, something is wrong"

        # this is where our boresight pointing should be to observe the passage.
        # this places our wafer set at the center of the source path, so the source
        # start at the edge of the array set and ends at the other edge.
        # note: bore + offset = array set center
        alt_bore = alt_center - alt_offset

        # get the time for source to reach the center
        # and use it as a reference to compute the scan direction phi_tilt
        t_center = utils.interp_extra(alt_center, alt, t)
        az_center = utils.interp_extra(t_center, t, az)
        daz_center = utils.interp_extra(t_center, t[1:], np.diff(az))
        dalt_center = utils.interp_extra(t_center, t[1:], np.diff(alt))
        # cos factor undo the projection effect
        phi_tilt = np.arctan2(daz_center * np.cos(np.deg2rad(alt_center)), dalt_center)

        # get the az bounds, since source is moving in a tilted path, we need to
        # find projected bounds (bounding box of the tilted path)
        bounds_x = np.array([
            inst.get_bounds_x_tilted(
                **spec,
                shape=self.spec_shape,
                phi_tilt=phi_tilt,
            ) for spec in self.specs])
        x_lo, x_hi = np.min(bounds_x[:, 0]), np.max(bounds_x[:, 1])
        # add back the projection effect to get the actual az bounds
        stretch = 1 / np.cos(np.deg2rad(alt_center))
        az_throw  = (x_hi - x_lo) * stretch  # diff by 2 from ACT convention
        az_offset = (x_hi + x_lo) / 2 * stretch
        az_bore   = az_center - az_offset

        # get validity ranges
        ok *= utils.within_bound(alt_stop, [alt.min(), alt.max()])
        if self.bounds_alt is not None:
            ok *= utils.within_bound(alt_bore, self.bounds_alt)
        if self.bounds_az_throw is not None:
            ok *= utils.within_bound(az_throw, self.bounds_az_throw)
        if self.max_obs_length is not None:
            ok *= (obs_length <= self.max_obs_length)
        ranges = utils.mask2ranges(ok)
        blocks = []
        for i_l, i_r in ranges:
            block = src.ObservingWindow(
                t0=utils.ct2dt(t[i_l]), t1=utils.ct2dt(t[i_r-1]), 
                name=block.name, mode=block.mode,
                t_start=t[i_l:i_r], obs_length=obs_length[i_l:i_r],
                az_bore=az_bore[i_l:i_r], alt_bore=alt_bore[i_l:i_r],
                az_throw=az_throw[i_l:i_r]
            )
            blocks.append(block)
        # probably the dominant mode of operation
        if len(blocks) == 1: return blocks[0]

    def applicable(self, block):
        return isinstance(block, src.SourceBlock)

@dataclass(frozen=True)
class SunAvoidance(MappableRule):
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

    def apply_block(self, block: core.Block) -> core.Blocks:
        """Calculate the distance between a block and a sun block."""
        sun_block = src.block_get_matching_sun_block(block)
        t, az_sun, alt_sun = sun_block.get_az_alt(time_step = dt.timedelta(seconds=self.time_step))

        if isinstance(block, inst.ScanBlock):
            daz, dalt = ((block.az - az_sun) + 180) % 360 - 180, block.alt - alt_sun
            daz, dalt = np.abs(daz) - block.throw, np.abs(dalt)
        elif isinstance(block, src.SourceBlock):  # no throw for source blocks
            az_interp, alt_interp = block.get_az_alt_interpolators()
            az, alt = az_interp(t), alt_interp(t)
            daz, dalt = ((az - az_sun) + 180) % 360 - 180, alt - alt_sun
            daz, dalt = np.abs(daz), np.abs(dalt)
        else:
            raise ValueError("Unknown block type")
        ok = np.logical_or(daz > self.min_angle_az, dalt > self.min_angle_alt)
        safe_intervals = utils.ranges_complement(utils.ranges_pad(utils.mask2ranges(~ok), self.n_buffer, len(az_sun)), len(az_sun))
        if len(safe_intervals) == 0: return None
        # if the whole block is safe, return it
        if np.all(safe_intervals[0] == [0, len(az_sun)]): return block
        # otherwise, split it up into safe intervals
        return [block.replace(t0=utils.ct2dt(t[i0]), t1=utils.ct2dt(t[i1-1])) for i0, i1 in safe_intervals]

    def applicable(self, block: core.Block) -> bool:
        return isinstance(block, (inst.ScanBlock, src.SourceBlock))

@dataclass(frozen=True)
class MakeSourceScan(MappableRule):
    """convert observing window to actual scan blocks and allow for
    rephasing of the block. Applicable to only ObservingWindow blocks.
    """
    preferred_length: float  # seconds
    rng_key: utils.PRNGKey

    def apply_block(self, block: core.Block) -> core.Block:
        duration = block.duration.total_seconds()
        preferred_len = min(self.preferred_length, duration)
        allowance = duration - preferred_len
        offset = utils.uniform(self.rng_key, 0, allowance)
        t0 = block.t0 + dt.timedelta(seconds=offset)
        return block.get_scan_starting_at(t0)

    def applicable(self, block: core.Block) -> bool:
        return isinstance(block, src.ObservingWindow)

# global registry of rules
RULES = {
    'alt-range': AltRange,
    'day-mod': DayMod,
    'drift-mode': DriftMode,
    'min-duration': MinDuration,
    'rephase-first': RephaseFirst,
    'sun-avoidance': SunAvoidance,
    'make-source-plan': MakeSourcePlan,
    'make-source-scan': MakeSourceScan,
}
def get_rule(name: str) -> core.Rule:
    return RULES[name]

def make_rule(name: str, **kwargs) -> core.Rule:
    assert name in RULES, f"unknown rule {name}"
    return get_rule(name)(**kwargs)
