from datetime import datetime, timezone
import pandas as pd
import numpy as np
from . import core, utils as u

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

def parse_sequence_from_toast(ifile: str) -> core.Blocks:
    """
    Parameters
    ----------
    ifile: input master schedule from toast
    """
    columns = ["start_utc", "stop_utc", "start_mjd", "stop_mjd",
               "rotation", "patch", "az_min", "az_max", "el", "mode"]
    df = pd.read_fwf(ifile, skiprows=3, header=None, index_col=None,
                     colspecs=[(0,20),(20,41),(41,57),(57,72),(72,81),
                               (81,116), (116,126), (126,135),(135,144),(144,146)], names=columns)
    blocks = []
    for _, row in df.iterrows():
        block = core.ScanBlock(
            t0=u.str2datetime(row['start_utc']),
            t1=u.str2datetime(row['stop_utc']),
            alt=row['el'],
            az=row['az_min'],
            throw=np.abs(row['az_max'] - row['az_min']),
            patch=row['patch']
        )
        blocks.append(block)

    return blocks

