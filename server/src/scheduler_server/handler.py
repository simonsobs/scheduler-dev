from datetime import datetime
from pathlib import Path

from .utils import split_into_parts
import schedlib as sl
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


def basic_policy(t0, t1, policy_config, app_config={}):
    policy = BasicPolicy(**policy_config)

    seq = policy.init_seqs(t0, t1)
    seq = policy.apply(seq)

    # convert to commands
    cmd = sl.seq2cmd(seq)
    return str(cmd)
