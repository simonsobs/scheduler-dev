from datetime import datetime
from pathlib import Path

import schedlib.utils
from .utils import split_into_parts, rand_upto
from schedlib.policies import BasicPolicy
import random

random.seed(int(datetime.now().timestamp()))
default_schedule = Path(__file__).parent / "schedule_sat.txt"

def dummy_policy(t0, t1, policy_config={}, app_config={}):
    dt = abs(t1.timestamp() - t0.timestamp())  # too lazy to check for t1<t0 now
    # get current time as a timestamp string
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = random.randint(1, 10)
    parts = split_into_parts(dt, n)
    commands = ["import time", f"# {now}"]
    for i, part in enumerate(parts):
        commands += [f"time.sleep({part:.2f})"]
    commands = "\n".join(commands)
    return commands


def basic_policy(t0, t1, policy_config={}, config={}):
    import schedlib as sl

    config.update(policy_config)
    policy = BasicPolicy.from_config(config)

    # get root path of the repo
    sat_schedule_ifile = policy.get("master_schedule", default_schedule)
    seq = schedlib.utils.parse_sequence_from_toast(sat_schedule_ifile)

    # filter out the observations that are not in the time range
    seq = sl.seq_filter(lambda x: x.t1 > t0, seq)
    seq = sl.seq_filter(lambda x: x.t0 < t1, seq)

    # make partial observations if needed
    if seq[0].t0 < t0:
        block = sl.block_shrink_left(seq[0], rand_upto(policy.get_block_tolerance(seq[0])))
        seq = [block] + seq[1:]

    # convert to commands
    cmd = sl.seq2cmd(seq)
    return str(cmd)
