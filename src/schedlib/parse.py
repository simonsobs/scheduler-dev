import pandas as pd
import numpy as np

from .core import Sequence, ScanBlock
from . import utils as u

def parse_sequence_from_toast(ifile: str) -> Sequence:
    """
    Parameters
    ----------
    ifile: input master schedule from toast

    Returns
    -------
    - Sequence object

    """
    # load schedules
    columns = ["start_utc", "stop_utc", "start_mjd", "stop_mjd",
               "rotation", "patch", "az_min", "az_max", "el", "mode"]
    df = pd.read_fwf(ifile, skiprows=3, header=None, index_col=None,
                     colspecs=[(0,20),(20,41),(41,57),(57,72),(72,81),
                               (81,116), (116,126), (126,135),(135,144),(144,146)], names=columns)
    blocks = []
    for _, row in df.iterrows():
        block = ScanBlock(
            t0=u.str2datetime(row['start_utc']),
            t1=u.str2datetime(row['stop_utc']),
            alt=row['el'],
            az=row['az_min'],
            throw=np.abs(row['az_max'] - row['az_min']),
            patch=row['patch']
        )
        blocks.append(block)

    return Sequence.from_blocks(blocks)
