from schedlib.policies import SATP3Policy
import datetime as dt

today = dt.date.today()
t0 = dt.datetime(today.year, today.month, today.day, 6, 0, 0, tzinfo=dt.timezone.utc)
t1 = t0 + dt.timedelta(days=1)

master_file = 'E60_A40_S2023-12-01_F2025-01-01_D-10_-40_L0.86_6.86_12.86_18.86_T65.00_M045_S045.txt'

policy = SATP3Policy.from_defaults(
    master_file = master_file,
    az_speed = 0.5,
    az_accel = 0.25,
    disable_hwp = False,
    apply_boresight_rot = False,
)

seqs = policy.build_schedule(t0, t1)
fname = 'schedule'
with open(fname, 'w') as f:
    f.write(seqs)
