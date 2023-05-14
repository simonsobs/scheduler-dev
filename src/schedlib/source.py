#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chex import dataclass
import ephem
import datetime as dt
from typing import Union, Callable, NamedTuple, List, Tuple
from scipy.interpolate import interp1d

from . import core, utils

class Location(NamedTuple):
    lat: float
    lon: float
    elev: float

    def at(self, date: dt.datetime) -> ephem.Observer:
        """Always get new object to avoid side effects."""
        obs = ephem.Observer()
        obs.lat = str(self.lat)
        obs.lon = str(self.lon)
        obs.elev = self.elev
        obs.date = ephem.date(date)
        return obs

DEFAULT_SITE = Location(lat=-22.958, lon=-67.786, elev=5200)

def get_site() -> Location:
    return DEFAULT_SITE


# source needs to be callable to avoid side effects
SOURCES = {
    'sun': ephem.Sun,
    'moon': ephem.Moon,
    'mercury': ephem.Mercury,
    'venus': ephem.Venus,
    'mars': ephem.Mars,
    'jupiter': ephem.Jupiter,
    'saturn': ephem.Saturn,
    'uranus': ephem.Uranus,
    'neptune': ephem.Neptune,
}

Source = Union[ephem.Body, ephem.FixedBody]

def get_source(name: str) -> Source:
    # always get new object to avoid side effects
    return SOURCES[name]()

def source_get_az_alt(source: str, times: List[dt.datetime]) -> core.Blocks:
    """Get altitude and azimuth for a source in a given list of times"""
    observer = get_site().at(times[0])
    source = get_source(source)
    az, alt = [], []
    for t in times:
        observer.date = ephem.date(t)
        source.compute(observer)
        az.append(source.az)
        alt.append(source.alt)
    return az, alt

def source_az_alt_interpolators(source: str, t0: dt.datetime, t1: dt.datetime, time_step: dt.timedelta):
    times = [t0 + i * dt for i in range(int((t1 - t0) / time_step))]
    alt, az = source_get_az_alt(source, times)
    times = [int(t.timestamp()) for t in times]
    interp_az = interp1d(times, az)
    interp_alt = interp1d(times, alt)
    return interp_az, interp_alt

@dataclass(Frozen=True)
class SourceBlock(core.Block):
    source: str
    mode: str
    def __post_init__(self):
        assert self.mode in ["rising", "setting"]

def source_get_blocks(source: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """Get altitude and azimuth for a source and save an interpolator.
    If interpolation functions are not available, build them."""
    # past is not as important as future
    site = get_site()
    source = get_source(source)
    t_block_beg = site.at(t0).previous_rising(source).datetime()
    t_block_mid = site.at(t_block_beg).next_transit(source).datetime()
    t_block_end = site.at(t_block_mid).next_setting(source).datetime()
    blocks = [SourceBlock(t0=t_block_beg, t1=t_block_mid, source=source, mode="rising"),
              SourceBlock(t0=t_block_mid, t1=t_block_end, source=source, mode="setting")]
    while t_block_end < t1:
        t_block_beg = site.at(t_block_end).next_rising(source).datetime()
        t_block_mid = site.at(t_block_beg).next_transit(source).datetime()
        t_block_end = site.at(t_block_mid).next_setting(source).datetime()
        blocks += [SourceBlock(t0=t_block_beg, t1=t_block_mid, source=source, mode="rising"),
                   SourceBlock(t0=t_block_mid, t1=t_block_end, source=source, mode="setting")]
    return blocks

# global registry of precomputed sources
PRECOMPUTED_SOURCES = {}

class PrecomputedSource(NamedTuple):
    t0: dt.datetime
    t1: dt.datetime
    interp_az: Callable[[int], float]
    interp_alt: Callable[[int], float]
    blocks: core.Blocks

    @classmethod
    def for_(cls, name: str, t0: dt.datetime, t1: dt.datetime,
             buf: dt.timedelta = dt.timedelta(days=7)):
        reuse = False
        if name in PRECOMPUTED_SOURCES:
            precomputed = PRECOMPUTED_SOURCES[name]
            reuse = precomputed.t0 <= t0 and precomputed.t1 >= t1
        if not reuse:
            # future is more important than past
            t0, t1 = t0, t1 + buf
            az_interp, alt_interp = source_az_alt_interpolators(name, t0, t1, dt.timedelta(seconds=1))
            blocks = source_get_blocks(name, t0, t1)
            PRECOMPUTED_SOURCES[name] = cls(t0, t1, az_interp, alt_interp, blocks)
        return PRECOMPUTED_SOURCES[name]

    @classmethod
    def for_block(cls, block: SourceBlock):
        return cls.for_(block.source, block.t0, block.t1)

def source_gen_seq(source: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """Get source blocks for a given source and time interval."""
    blocks = PrecomputedSource.for_(source, t0, t1).blocks
    return core.seq_trim(blocks, t0, t1)

def source_block_get_az_alt(block: SourceBlock, time_step: dt.timedelta) -> Tuple[List[dt.datetime], core.Array[float], core.Array[float]]:
    """Get altitude and azimuth for a source block."""
    source = PrecomputedSource.for_block(block)
    t0, t1 = block.t0, block.t1
    times = [t0 + i * time_step for i in range(int((t1 - t0) / time_step))]
    ctimes = [int(t.timestamp()) for t in times]
    az = source.interp_az(ctimes)
    alt = source.interp_alt(ctimes)
    return times, az, alt

def source_block_trim_by_alt_range(block:SourceBlock, alt_range:Tuple[float, float]) -> core.Blocks:
    times, _, alt = source_block_get_az_alt(block)
    alt_min, alt_max = alt_range
    mask = (alt_min <= alt) * (alt <= alt_max)
    if not mask.any():
        return None
    blocks = []
    for (i0, i1) in utils.mask2ranges(mask): 
        t0 = times[i0]
        t1 = times[i1-1]  # i1 is non-inclusive
        blocks.append(SourceBlock(t0=t0, t1=t1, source=block.source, mode=block.mode))
    return blocks