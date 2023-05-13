from datetime import datetime
import random

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

def split_into_parts(N, m):
    parts = []
    for i in range(m-1):
        parts.append(random.uniform(0, N/m))
        N -= parts[-1]
    parts.append(N)
    random.shuffle(parts)
    return parts

def basic_policy(t0, t1, policy_config={}, config={}):
    import schedlib as sl
    from pathlib import Path
    config.update(policy_config)

    # get root path of the repo
    default_schedule = Path(__file__).parent / "schedule_sat.txt"
    sat_schedule_ifile = config.get("master_schedule", default_schedule)
    seq = sl.parse_sequence_from_toast(sat_schedule_ifile)

    # filter out the observations that are not in the time range
    seq = sl.seq_filter(lambda x: x.t1 > t0, seq)
    seq = sl.seq_filter(lambda x: x.t0 < t1, seq)

    # make partial observations if needed
    # sl.seq_map_when(lambda x: (x.t0 < t0) and (x.t1 - t0))

    # convert to commands
    cmd = sl.seq2cmd(seq)
    return str(cmd)