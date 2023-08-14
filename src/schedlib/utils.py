from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from functools import reduce
from contextlib import contextmanager
from typing import Any, List
from scipy import interpolate
from collections.abc import Iterable

from . import core, utils as u, instrument as inst


minute = 60 # second
hour = 60 * minute
day = 24 * hour
sidereal_day = 0.997269566 * day
deg = np.pi / 180

def str2ctime(time_str):
    ctime = (pd.Timestamp(time_str) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return ctime

def str2datetime(time_str):
    ctime = str2ctime(time_str)
    return datetime.fromtimestamp(int(ctime), tz=timezone.utc)

def datetime2str(dtime):
    return dtime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

def ct2dt(ctime):
    if isinstance(ctime, Iterable):
        return [datetime.utcfromtimestamp(t).astimezone(timezone.utc) for t in ctime]
    else:
        try:
            return datetime.utcfromtimestamp(ctime).astimezone(timezone.utc)
        except TypeError:
            raise ValueError(f"ctime should be int, float or iterable, not {type(ctime)}")

def dt2ct(dtime):
    if isinstance(dtime, Iterable):
        return np.array([int(d.timestamp()) for d in dtime])
    else:
        try:
            return int(dtime.timestamp())
        except TypeError:
            raise ValueError(f"dtime should be datetime or iterable, not {type(dtime)}")

def mask2ranges(mask):
    # handle a bunch of special cases first
    if len(mask) == 0: return np.empty((0, 2), dtype=int)
    if np.all(mask): return np.array([[0, len(mask)]], dtype=int)
    if not np.any(mask): return np.empty((0, 2), dtype=int)
    # general case
    bounds = np.where(np.diff(mask) != 0)[0] + 1  # 1 to n-1
    bounds = np.concatenate(([0], bounds, [len(mask)]))  # 0 to n
    bounds = np.vstack((bounds[:-1], bounds[1:])).T # [[(0 to n-1), (1 to n)], ...]
    return bounds[mask[bounds[:, 0]] == 1]

def ranges2mask(ranges, imax):
    mask = np.zeros(imax, dtype=bool)
    for i_left, i_right in ranges:
        mask[i_left:i_right] = True
    return mask

def ranges_pad(ranges, pad, imax):
    """pad each range and merge overlapping ranges"""
    ranges = ranges + np.array([-pad, pad])
    ranges = np.clip(ranges, 0, imax)
    if len(ranges) < 2: return ranges
    # merge overlapping ranges
    return np.array(reduce(lambda l, r: l[:-1] + [[l[-1][0], r[1]]] if l[-1][-1] >= r[0] else l + [r],
                           ranges[1:], [ranges[0]]))

def ranges_complement(ranges, imax):
    """return the complement ranges"""
    return mask2ranges(~ranges2mask(ranges, imax))

def parse_sequence_from_toast(ifile: str, verbose=False) -> core.Blocks:
    """
    Parameters
    ----------
    ifile: input master schedule from toast
    verbose: whether toast schedule is produced from verbose mode
    """
    if verbose:
        columns = ["start_utc", "stop_utc", "start_mjd", "stop_mjd",
                   "rotation", "patch", "az_min", "az_max", "el", "mode"]
        df = pd.read_fwf(ifile, skiprows=3, header=None, index_col=None,
                         colspecs=[(0,20),(20,41),(41,57),(57,72),(72,81),
                                   (81,116), (116,126), (126,135),(135,144),(144,146)], names=columns)
    else:
        columns = ["start_utc", "stop_utc", "rotation", "patch", "az_min", "az_max", "el", "pass", "sub"]
        df = pd.read_fwf(ifile, skiprows=3, header=None, index_col=None,
                         colspecs=[(0,21),(21,42),(42,51),(51,89),(89,96),
                                   (96,105), (105,114), (114,120),(120,125)], names=columns)
    blocks = []
    for _, row in df.iterrows():
        block = inst.ScanBlock(
            name=row['patch'],
            t0=u.str2datetime(row['start_utc']),
            t1=u.str2datetime(row['stop_utc']),
            alt=row['el'],
            az=row['az_min'],
            throw=np.abs(row['az_max'] - row['az_min']),
        )
        blocks.append(block)

    return blocks

# convenience wrapper for interpolation: numpy-like scipy interpolate
def interp_extra(x_new, x, y):
    """interpolate with extrapolation"""
    return interpolate.interp1d(x, y, fill_value='extrapolate', bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def interp_bounded(x_new, x, y):
    """interpolate with bounded extrapolation"""
    return interpolate.interp1d(x, y, fill_value=(y[0], y[-1]), bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def within_bound(x: core.Arr[Any], bounds: List[float]) -> core.Arr[bool]:
    """return a boolean mask indicating whether x is within the bound"""
    return (x >= bounds[0]) * (x <= bounds[1])

# ====================
# Random utilities
# ====================
# In order to produce reproducible schedules, we need to be careful
# about random number generator state, especially if the service runs
# in a server setting, because the state of our random number will
# then depends on the order of requests received which is not desirable
# for reproducibility. We use the following pattern to ensure
# reproducibility: always use these wrapper functions to generate
# random numbers, and always pass a PRNGKey object to them.
# It should be obvious where the inspirataion comes from.

class PRNGKey:
    """motivation: reduce side effects: same key should always produce same
    result and not affect other keys"""
    def __init__(self, key):
        self.key = key  # need to be hashable
        self.state = None

    @contextmanager
    def set_state(self):
        old_state = np.random.get_state()
        try:
            if self.state is None:
                seed = hash(self.key) % (2**32 - 1)
                np.random.seed(seed)
                self.state = np.random.get_state()
            else:
                np.random.set_state(self.state)
            yield
        finally:
            np.random.set_state(old_state)

    def split(self, n=2):
        """split the key into n keys. Used tuple to avoid collisions"""
        return [PRNGKey((self.key, i)) for i in range(n)]

# now we can make some wrappers for common numpy.random functions
def uniform(key: PRNGKey, low=0.0, high=1.0, size=None):
    with key.set_state():
        return np.random.uniform(low, high, size)

def daily_static_key(t: datetime):
    return PRNGKey((t.year, t.month, t.day))

def pprint(seq: core.BlocksTree):
    """pretty print"""
    from equinox import tree_pprint
    tree_pprint(seq)
