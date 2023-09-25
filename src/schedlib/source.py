# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import ephem
from ephem import to_timezone
import datetime as dt
from typing import Union, Callable, NamedTuple, List, Tuple, Optional
from scipy.interpolate import interp1d
import numpy as np

from . import core, utils, instrument as inst

UTC = dt.timezone.utc

class Location(NamedTuple):
    """Location given in degrees and meters"""
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

def _source_get_az_alt(source: str, times: List[dt.datetime]):
    """Get altitude and azimuth for a source in a given list of times"""
    assert len(times) > 0, "Need at least one time"
    if times[0].tzinfo is None:
        raise ValueError("Need timezone-aware datetime")
    observer = get_site().at(times[0])
    source = get_source(source)
    az, alt = [], []
    for t in times:
        observer.date = ephem.date(t)
        source.compute(observer)
        az.append(np.rad2deg(source.az))
        alt.append(np.rad2deg(source.alt))
    az = np.unwrap(np.array(az), period=360)
    alt = np.array(alt)
    return az, alt

def _source_az_alt_interpolators(source: str, t0: dt.datetime, t1: dt.datetime, time_step: dt.timedelta):
    times = [t0 + i * time_step for i in range(int((t1 - t0) / time_step))]
    az, alt = _source_get_az_alt(source, times)
    times = [int(t.timestamp()) for t in times]
    interp_az = interp1d(times, az, kind='cubic')
    interp_alt = interp1d(times, alt, kind='cubic')
    return interp_az, interp_alt

# global registry of precomputed sources
PRECOMPUTED_SOURCES = {}

class _PrecomputedSource(NamedTuple):
    t0: dt.datetime
    t1: dt.datetime
    interp_az: Callable[[int], float]
    interp_alt: Callable[[int], float]
    blocks: core.Blocks

    @classmethod
    def for_(cls, name: str, t0: dt.datetime, t1: dt.datetime,
             buf: dt.timedelta = dt.timedelta(days=1),
             time_step: dt.timedelta = dt.timedelta(seconds=30)) -> Source:
        reuse = False
        if name in PRECOMPUTED_SOURCES:
            precomputed = PRECOMPUTED_SOURCES[name]
            reuse = precomputed.t0 <= t0 and precomputed.t1 >= t1
        if not reuse:
            # future is more important than past
            t0, t1 = t0, t1 + buf
            az_interp, alt_interp = _source_az_alt_interpolators(name, t0, t1, time_step)
            blocks = source_get_blocks(name, t0, t1)
            PRECOMPUTED_SOURCES[name] = cls(t0, t1, az_interp, alt_interp, blocks)
        return PRECOMPUTED_SOURCES[name]

    @classmethod
    def for_block(cls, block: SourceBlock, buf: dt.timedelta = dt.timedelta(days=1),
                  time_step: dt.timedelta = dt.timedelta(seconds=30)) -> Source:
        return cls.for_(block.name, block.t0, block.t1, buf=buf, time_step=time_step)

@dataclass(frozen=True)
class SourceBlock(core.NamedBlock):
    mode: str
    def __post_init__(self):
        if not self.mode in ["rising", "setting", "both"]:
            raise ValueError("mode must be rising or setting or both")
    def get_az_alt(self, time_step: dt.timedelta = dt.timedelta(seconds=30)) -> Tuple[List[dt.datetime], core.Arr, core.Arr]:
        """Return times, az, alt for a source block at a given time step"""
        return source_block_get_az_alt(self, time_step)
    def trim_by_az_alt_range(self, az_range: Optional[Tuple[float, float]] = None,
                             alt_range: Optional[Tuple[float, float]] = None,
                             time_step: dt.timedelta = dt.timedelta(seconds=30)):
        """Trim a source block by azimuth and altitude ranges"""
        return source_block_trim_by_az_alt_range(self, az_range, alt_range, time_step)
    @property
    def az(self):
        return self.get_az_alt()[1]
    @property
    def alt(self):
        return self.get_az_alt()[2]
    def get_az_alt_interpolators(self):
        source = _PrecomputedSource.for_block(self)
        return source.interp_az, source.interp_alt

def source_get_blocks(name: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """Get altitude and azimuth for a source and save an interpolator.
    If interpolation functions are not available, build them."""
    site = get_site()
    source = get_source(name)
    t_block_beg = to_timezone(site.at(t0).previous_rising(source), UTC)
    t_block_mid = to_timezone(site.at(t_block_beg).next_transit(source), UTC)
    t_block_end = to_timezone(site.at(t_block_mid).next_setting(source), UTC)
    blocks = [SourceBlock(t0=t_block_beg, t1=t_block_mid, name=name, mode="rising"),
              SourceBlock(t0=t_block_mid, t1=t_block_end, name=name, mode="setting")]
    while t_block_end < t1:
        t_block_beg = to_timezone(site.at(t_block_end).next_rising(source), UTC)
        t_block_mid = to_timezone(site.at(t_block_beg).next_transit(source), UTC)
        t_block_end = to_timezone(site.at(t_block_mid).next_setting(source), UTC)
        blocks += [SourceBlock(t0=t_block_beg, t1=t_block_mid, name=name, mode="rising"),
                   SourceBlock(t0=t_block_mid, t1=t_block_end, name=name, mode="setting")]
    return blocks

def source_gen_seq(source: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """similar to source_get_blocks but it will try loading from cache first and will
    return a trimmed sequence of blocks."""
    blocks = _PrecomputedSource.for_(source, t0, t1).blocks
    return core.seq_flatten(core.seq_trim(blocks, t0, t1))

def source_block_get_az_alt(block: SourceBlock, time_step: dt.timedelta = dt.timedelta(seconds=30)) -> Tuple[core.Arr[int], core.Arr[float], core.Arr[float]]:
    """Get altitude and azimuth for a source block."""
    source = _PrecomputedSource.for_block(block, time_step=time_step)
    t0, t1 = block.t0, block.t1
    times = [t0 + i * time_step for i in range(int((t1 - t0) / time_step))]
    ctimes = np.array([int(t.timestamp()) for t in times])
    az = source.interp_az(ctimes)
    alt = source.interp_alt(ctimes)
    return ctimes, az, alt

def source_block_trim_by_az_alt_range(block: SourceBlock, az_range:Optional[Tuple[float, float]]=None, alt_range:Optional[Tuple[float, float]]=None, time_step:dt.timedelta=dt.timedelta(seconds=30)) -> core.Blocks:
    """alt_range: (alt_min, alt_max) in radians
    az_range: (az_min, az_max) in radians"""
    if az_range is None and alt_range is None:
        # not sure why one would want to do this though
        return [block]
    times, az, alt = source_block_get_az_alt(block)
    mask = np.ones_like(az, dtype=bool)
    if az_range is not None:
        az_min, az_max = az_range
        mask *= (az_min <= az) * (az <= az_max)
    if alt_range is not None:
        alt_min, alt_max = alt_range
        mask *= (alt_min <= alt) * (alt <= alt_max)
    if not mask.any():
        return []  # need blocks type
    blocks = []
    for (i0, i1) in utils.mask2ranges(mask): 
        t0 = utils.ct2dt(times[i0])
        t1 = utils.ct2dt(times[i1-1])  # i1 is non-inclusive
        blocks.append(block.replace(t0=t0, t1=t1))
    return blocks

def block_get_matching_sun_block(block: core.Block) -> SourceBlock:
    """get the corresponding sun block for a given block with
    the same time bounds."""
    return SourceBlock(t0=block.t0, t1=block.t1, name="sun", mode="both")

@dataclass(frozen=True)
class ObservingWindow(SourceBlock):
    t_start: core.Arr[float]
    obs_length: core.Arr[float]
    az_bore: core.Arr[float]
    alt_bore: core.Arr[float]
    az_throw: core.Arr[float]
    def get_scan_starting_at(self, t0: dt.datetime) -> inst.ScanBlock:
        """get a possible scan starting at t0"""
        t_req = int(t0.timestamp())
        # if we start at t0, we can observe for at most obs_length
        obs_length = utils.interp_bounded(t_req, self.t_start, self.obs_length)
        t1 = t0 + dt.timedelta(seconds=float(obs_length))
        # if we start at t0, we can observe with these parameters
        az = utils.interp_bounded(t_req, self.t_start, self.az_bore)
        alt = utils.interp_bounded(t_req, self.t_start, self.alt_bore)
        az_throw = utils.interp_bounded(t_req, self.t_start, self.az_throw)
        return inst.ScanBlock(
            name=self.name,
            t0=t0,
            t1=t1,
            az=float(az),
            alt=float(alt),
            throw=float(az_throw),
        )
    def get_scan_at_alt(self, alt: float) -> inst.ScanBlock:
        """get a possible scan at a given altitude"""
        t0 = utils.interp_bounded(alt, self.alt_bore, self.t_start)
        return self.get_scan_starting_at(t0)
