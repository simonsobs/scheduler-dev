from datetime import datetime
import random

def dummy_policy(t0, t1, config={}):
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

def split_into_parts(N, m):
    parts = []
    for i in range(m-1):
        parts.append(random.uniform(0, N/m))
        N -= parts[-1]
    parts.append(N)
    random.shuffle(parts)
    return parts

def basic_policy(t0, t1, config={}):
    from schedlib import parse_sequence_from_toast, Sequence, sequence2command
    from pathlib import Path

    # get root path of the repo
    default_schedule = Path(__file__).parent / "schedule_sat.txt"
    sat_schedule_ifile = config.get("master_schedule", default_schedule)
    seq = parse_sequence_from_toast(sat_schedule_ifile)

    # filter out the observations that are not in the time range
    seq = Sequence(blocks=[b for b in seq.blocks if t0 <= b.t0 <= t1])
    cmd = sequence2command(seq)
    return str(cmd)