from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from functools import reduce
from contextlib import contextmanager
from scipy import interpolate
from collections.abc import Iterable
from jax.tree_util import SequenceKey, DictKey
import fnmatch

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
    if isinstance(ctime, list):
        return [datetime.fromtimestamp(t).astimezone(timezone.utc) for t in ctime]
    else:
        try:
            return datetime.fromtimestamp(ctime).astimezone(timezone.utc)
        except TypeError:
            raise ValueError(f"ctime should be int, float or iterable, not {type(ctime)}")

def dt2ct(dtime):
    if isinstance(dtime, Iterable):
        return np.array([float(d.timestamp()) for d in dtime])
    else:
        try:
            return float(dtime.timestamp())
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


# convenience wrapper for interpolation: numpy-like scipy interpolate
def interp_extra(x_new, x, y, fill_value='extrapolate'):
    """interpolate with extrapolation"""
    return interpolate.interp1d(x, y, fill_value=fill_value, bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def interp_bounded(x_new, x, y):
    """interpolate with bounded extrapolation"""
    return interpolate.interp1d(x, y, fill_value=(y[0], y[-1]), bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def within_bound(x, bounds):
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

def pprint(seq):
    """pretty print"""
    from equinox import tree_pprint
    tree_pprint(seq)

# ====================
# path related
# ====================

def path2key(path, ignore_seqkey=False):
    """convert a path (used in tree_util.tree_map_with_path) to a dot-separated key
    
    Parameters
    ----------
    path: a list of SequenceKey or DictKey
    ignore_array: if True, ignore the SequencyKey index in the path 

    Returns
    -------
    key: a string of dot-separated keys

    """
    keys = []
    for p in path:
        if isinstance(p, SequenceKey):
            if not ignore_seqkey:
                keys.append(p.idx)
        elif isinstance(p, DictKey):
            keys.append(p.key)
        else:
            raise ValueError(f"unknown path type {type(p)}")
    return ".".join([str(k) for k in keys])

def match_query(path, query):
    """in order for a query to match with a path, it can
    satisfy the following: 
    1. the query is a substring of the path
    2. the query is a glob pattern that matches the path
    3. if the query is a comma-separated list of multiple queries, 
    any of them meeting comdition 1 and 2 will return True
    """
    key = path2key(path)
    # first match the constraint to key
    queires = query.split(",")
    for q in queires:
        if q in key: 
            return True
        if fnmatch.fnmatch(key, q): 
            return True
    return False

def nested_update(dictionary, update_dict, new_keys_allowed=True):
    """update a nested dictionary recursively but
    never add new keys"""
    if update_dict is None: return dictionary
    for key, value in update_dict.items():
        if key in dictionary and isinstance(dictionary[key], dict) and isinstance(value, dict):
            nested_update(dictionary[key], value)
        else:
            # optional to prevent new keys to be added with new_keys_allowed=False
            if key in dictionary or new_keys_allowed:
                dictionary[key] = value
    return dictionary