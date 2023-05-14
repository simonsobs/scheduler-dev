from datetime import datetime, timezone
import pandas as pd
import numpy as np

def str2ctime(time_str):
    ctime = (pd.Timestamp(time_str) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return ctime

def str2datetime(time_str):
    ctime = str2ctime(time_str)
    return datetime.fromtimestamp(int(ctime), tz=timezone.utc)

def datetime2str(dtime):
    return dtime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

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
