# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import ephem
from ephem import to_timezone
import datetime as dt
from typing import Union, Callable, NamedTuple, List, Tuple, Optional
from scipy.interpolate import interp1d
import numpy as np
from scipy import interpolate
from so3g.proj import quat

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
    def get_scan_at_t0(self, t0: dt.datetime) -> inst.ScanBlock:
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
        return self.get_scan_at_t0(t0)

def make_source_ces(block, array_info, el_bore=50, drift_params=None, enable_drift=False, verbose=False):
    assert 'center' in array_info and 'cover' in array_info
    # move to the frame in which the center of the wafer is at the origin
    q_center = quat.rotation_xieta(*array_info['center'])
    q_cover = quat.rotation_xieta(*array_info['cover'])
    xi_cover_array, eta_cover_array, _ = quat.decompose_xieta(~q_center * q_cover)
    # find out the elevation of the array if boresight is at el_bore
    _, dalt, _ = quat.decompose_lonlat(quat.rotation_lonlat(0, 0) * q_center)
    el_array = el_bore + dalt / utils.deg
    # get trajectory of the source
    t, az_src, el_src = block.get_az_alt()  # degs
    if drift_params is not None:
        assert 't' in drift_params and 'az_speed' in drift_params
        v_az = drift_params['az_speed']
        az_src -= (t - drift_params['t']) * drift_params['az_speed']
    else:
        v_az = 0
    # # az of the source when el_src = el_array
    if el_array > max(el_src):
        return None
    if el_array < min(el_src):
        return None
    az_array = interpolate.interp1d(el_src, az_src)(el_array)
    # center array on the source and put it at the origin
    q_src_ground = quat.rotation_lonlat(-az_src * utils.deg, el_src * utils.deg)
    q_target_ground = quat.rotation_lonlat(-az_array * utils.deg, el_array * utils.deg)
    q_src_array = ~q_target_ground * q_src_ground  # where target is at the origin
    xi_src_array, eta_src_array, _ = quat.decompose_xieta(q_src_array)
    # make sure a scan of the entire array is possible
    if block.mode == 'rising':
        if max(eta_src_array) < max(eta_cover_array) or min(eta_src_array) > min(eta_cover_array): 
            return None
    if block.mode == 'setting':
        if min(eta_src_array) > min(eta_cover_array) or max(eta_src_array) < max(eta_cover_array): 
            return None
    # work out the tilt of the wafer at the origin
    phi_tilt_fun = interpolate.interp1d(eta_src_array[:-1], 
                                        np.arctan2(np.diff(xi_src_array), 
                                                   np.diff(eta_src_array)), 
                                        fill_value='extrapolate')
    # find out the boundaries of the wafer by taking a projection along the tilt axis
    x_cross = - eta_cover_array * np.tan(phi_tilt_fun(eta_cover_array)) + xi_cover_array
    q_A_array = quat.rotation_xieta(np.min(x_cross), 0)
    q_B_array = quat.rotation_xieta(np.max(x_cross), 0)
    az_A_array, _, _ = quat.decompose_lonlat(quat.rotation_lonlat(0, el_array * utils.deg) * q_A_array)
    az_A_array *= -1
    az_B_array, _, _ = quat.decompose_lonlat(quat.rotation_lonlat(0, el_array * utils.deg) * q_B_array)
    az_B_array *= -1 
    # now we have all ingradients to make a source scan
    daz, dalt, _ = quat.decompose_lonlat(quat.rotation_lonlat(0, el_bore * utils.deg) * quat.rotation_xieta(*array_info['center']))
    daz *= -1
    # az boresight should move between these two points
    az_A_bore = az_array * utils.deg + az_A_array - daz  # rad
    az_B_bore = az_array * utils.deg + az_B_array - daz  # rad
    # get scan az and throw
    az_start = min(az_A_bore, az_B_bore)  # rad
    throw = abs(az_B_bore - az_A_bore)    # rad
    q_bore_start = quat.rotation_lonlat(-az_start, el_bore * utils.deg)
    az_cover_start, el_cover_start, _ = quat.decompose_lonlat(q_bore_start * q_cover)
    az_cover_start *= -1
    # get the elevation ranges
    if block.mode == 'rising': 
        el_src_start = np.min(el_cover_start) / utils.deg
        el_src_stop = np.max(el_cover_start) / utils.deg
        if el_src_start < np.min(el_src) or el_src_stop > np.max(el_src):
            return None
    elif block.mode == 'setting':
        el_src_start = np.max(el_cover_start) / utils.deg
        el_src_stop = np.min(el_cover_start) / utils.deg
        if el_src_start > np.max(el_src) or el_src_stop < np.min(el_src):
            return None
    else:
        raise ValueError(f'unsupported scan mode encountered: {block.mode}')
    # get the time ranges
    t_start = interpolate.interp1d(el_src, t)(el_src_start)
    t_stop  = interpolate.interp1d(el_src, t)(el_src_stop)
    t0 = utils.ct2dt(float(t_start))
    t1 = utils.ct2dt(float(t_stop))
    if enable_drift:
        az_speed_ref = np.median(np.diff(az_src) / np.diff(t))
        drift_params = {'t': t_start, 'az_speed': az_speed_ref}
        return make_source_ces(block, array_info, el_bore=el_bore, drift_params=drift_params, enable_drift=False, verbose=verbose)
    else:
        if verbose:
            print("t0 = ", t_start)
            print("t1 = ", t_stop)
            print("az = ", az_start / utils.deg)
            print("throw = ", throw / utils.deg)
            print("drift = ", v_az)
        return inst.ScanBlock(name=block.name, az=az_start / utils.deg, alt=el_bore, throw=throw / utils.deg, t0=t0, t1=t1, drift=v_az)