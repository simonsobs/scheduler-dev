# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import ephem
from ephem import to_timezone
import datetime as dt
from typing import Union, Callable, NamedTuple, List, Tuple, Optional
import numpy as np
from scipy import interpolate, optimize
from so3g.proj import quat

from . import core, utils as u, instrument as inst

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

def _debabyl(deg, arcmin, arcsec):
    return deg + arcmin/60 + arcsec/3600

SITES = {
    'act':   Location(lat=-22.9585, lon=-67.7876, elev=5188),
    'lat':   Location(lat=-_debabyl(22,57,39.47), lon=-_debabyl(67,47,15.68), elev=5188),
    'satp1': Location(lat=-_debabyl(22,57,36.38), lon=-_debabyl(67,47,18.11), elev=5188),
    'satp2': Location(lat=-_debabyl(22,57,36.35), lon=-_debabyl(67,47,17.28), elev=5188),
    'satp3': Location(lat=-_debabyl(22,57,35.97), lon=-_debabyl(67,47,16.53), elev=5188),
}
DEFAULT_SITE = Location(lat=-22.958, lon=-67.786, elev=5200)

def get_site(site='lat') -> Location:
    """use lat as default following so3g convention"""
    return SITES[site]

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
    interp_az = interpolate.interp1d(times, az, kind='cubic')
    interp_alt = interpolate.interp1d(times, alt, kind='cubic')
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
    # make sure az is at a reasonable range
    az = np.unwrap(np.mod(az, 360), period=360)
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
    for (i0, i1) in u.mask2ranges(mask):
        t0 = u.ct2dt(times[i0])
        t1 = u.ct2dt(times[i1-1])  # i1 is non-inclusive
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
    def get_scan_at_t0(self, t0: dt.datetime) -> inst.ScanBlock:
        """get a possible scan starting at t0"""
        t_req = int(t0.timestamp())
        # if we start at t0, we can observe for at most obs_length
        obs_length = u.interp_bounded(t_req, self.t_start, self.obs_length)
        t1 = t0 + dt.timedelta(seconds=float(obs_length))
        # if we start at t0, we can observe with these parameters
        az = u.interp_bounded(t_req, self.t_start, self.az_bore)
        alt = u.interp_bounded(t_req, self.t_start, self.alt_bore)
        az_throw = u.interp_bounded(t_req, self.t_start, self.az_throw)
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
        t0 = u.interp_bounded(alt, self.alt_bore, self.t_start)
        return self.get_scan_at_t0(t0)

def _find_az_bore(el_bore, az_src, el_src, q_point, atol=0.01):
    """find the boresight, given el_bore, such that q_point (relative to the boresight) is
    intercepted by the trajectory of the source

    """
    def fun(az_bore):
        az_center, el_center, _ =  quat.decompose_lonlat(
            quat.rotation_lonlat(-az_bore * u.deg, el_bore * u.deg) * q_point
        )
        az_center *= -1
        az_expect = interpolate.interp1d(
            el_src, az_src, fill_value='extrapolate'
        )(el_center / u.deg)
        return np.mod(np.abs(az_expect - az_center / u.deg),360)
    az_bore_init = interpolate.interp1d(
        el_src, az_src, fill_value='extrapolate'
    )(el_bore)
    res = optimize.minimize(fun, az_bore_init, method='Nelder-Mead')
    assert res.success, 'failed to converge on where to point the boresight'
    az_bore = res.x[0]
    # extra check
    if fun(az_bore) > atol:
        raise ValueError(f"failed to meet convergence tol ({atol}) on where to point the boresight")
    return az_bore

def make_source_ces(block, array_info, el_bore=50, 
        allow_partial=False, v_az=None, boresight_rot=None
    ):
    """make a ces scan of a source

    Parameters
    ----------
    block: SourceBlock
        a source block to compute the ces scan for
    array_info: dict
        contains center and cover of the array
    el_bore: float
        elevation of the boresight in degrees
    allow_partial: bool
        if True, allow partial coverage of the array
    v_az: Optional[float]
        az drift speed in az in deg/s, if None, will try to find optimal drift speed
    boresight_rot: Optional[float]
        rotation of the boresight in deg

    Returns
    -------
    inst.ScanBlock
        a scan block that can be used to scan the source

    """
    assert 'center' in array_info and 'cover' in array_info, 'array_info must contain center and cover'
    q_center = quat.rotation_xieta(*array_info['center'])
    q_cover = quat.rotation_xieta(*array_info['cover'])

    # apply boresight rotation if specified
    if boresight_rot is not None:
        q_bore_rot = quat.euler(2, -np.deg2rad(boresight_rot))
        q_center = q_bore_rot * q_center
        q_cover = q_bore_rot * q_cover

    t, az_src, el_src = block.get_az_alt()  # degs
    t_src_interp = interpolate.interp1d(
        el_src, t, kind='linear', fill_value='extrapolate'
    )

    # work out boresight
    az_bore = _find_az_bore(el_bore, az_src, el_src, q_center)
    q_bore = quat.rotation_lonlat(-az_bore * u.deg, el_bore * u.deg)

    # put array on the sky
    az_cover, el_cover, _ = quat.decompose_lonlat(q_bore * q_cover)
    az_cover *= -1

    # can we cover the full array?
    if not allow_partial:
        if np.max(el_cover) / u.deg > np.max(el_src):
            print("Source will not cover the top part of the array")
            return None
        if np.min(el_cover) / u.deg < np.min(el_src):
            print("Source will not cover the bottom part of the array")
            return None

    if block.mode == 'rising':
        t0 = t_src_interp(max(np.min(el_cover) / u.deg, np.min(el_src)))
        t1 = t_src_interp(min(np.max(el_cover) / u.deg, np.max(el_src)))
    elif block.mode == 'setting':
        t0 = t_src_interp(min(np.max(el_cover) / u.deg, np.max(el_src)))
        t1 = t_src_interp(max(np.min(el_cover) / u.deg, np.min(el_src)))
    else:
        raise ValueError(f'unsupported scan mode encountered: {block.mode}')

    # now we have the time bounds, we will find approximate optimal drift
    def _find_approx_az_throw(az_drift, az_src, el_src):
        az_src = az_src - az_drift * (t - t0)
        az_src_interp = interpolate.interp1d(el_src, az_src, kind='cubic')
        # find the az distance to the source for each point on the array cover
        distances = []
        for az_, el_ in zip(az_cover / u.deg, el_cover / u.deg):
            # we'll only be here if allow_partial is True, in which case
            # we safely ignore these points
            if el_ > np.max(el_src) or el_ < np.min(el_src):
                continue
            distance = az_src_interp(el_) - az_
            distances.append(distance)
        distances = np.array(distances)
        if len(distances) == 0:
            print("Source will not cover the array at all")
            raise ValueError("Source will not cover the array at all")
        az0, az1 = np.array([np.min(distances), np.max(distances)]) + az_bore
        throw = az1 - az0
        return az0, throw

    # only solve if no az_drift are specified
    if v_az is None:
        try:
            res = optimize.minimize(lambda x, *args: _find_approx_az_throw(x, *args)[1], 0, args=(az_src, el_src), method='Nelder-Mead')
            if not res.success:
                raise ValueError("Failed to find optimal drift, using median az speed instead")
            else:
                v_az = res.x[0]
        except ValueError:
            print("Failed to find optimal drift, using median az speed instead")
            v_az = np.median(np.diff(az_src) / np.diff(t))
    try:
        az0, throw = _find_approx_az_throw(v_az, az_src, el_src)
        return inst.ScanBlock(
            name=block.name, 
            az=az0,
            alt=el_bore, 
            throw=throw,
            t0=u.ct2dt(float(t0)),
            t1=u.ct2dt(float(t1)),
            az_drift=v_az,
            boresight_angle=boresight_rot,
            tag=f"{block.name},{block.mode}"
        )
    except ValueError:
        print("Failed to find optimal drift, using median az speed instead")
        return None
