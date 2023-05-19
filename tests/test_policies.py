from schedlib import policies
import datetime as dt
import os.path as op
from equinox import tree_pprint

minute = 60
def test_basic_policy():
    config = {
        'master_schedule': op.dirname(__file__) + '/data/schedule_sat.txt',
        'rules': {
            'rephase-first': {
                'max_fraction': 0.1,
                'min_block_size': 600,  # in seconds
            },
            'sun-avoidance': {
                'min_angle_az': 6,
                'min_angle_alt': 6,
                'time_step': 30,   # sec
                'n_buffer': 10, # 30 * 10 = 5 mins
            },
            'day-mod': {
                'day': 0,
                'day_mod': 1,
                'day_ref': dt.datetime(2014, 1, 1, 0, 0, 0),
            },
            'min-duration-cal': {
                'min_duration': 20 * minute,
            },
        },
        'calibration_targets': ["uranus"],
        'soft_targets': ["saturn"],
    }
    policy = policies.BasicPolicy(**config)
    seqs = policy.init_seqs(
        dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 2, 1, 0, 0, tzinfo=dt.timezone.utc))
    # policy.apply(seqs)

