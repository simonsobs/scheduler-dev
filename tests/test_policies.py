from schedlib import policies
import datetime as dt
import os.path as op

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
            'drift-mode': {
                'mode': 'rising'
            },
            'min-duration-cal': {
                'min_duration': 5 * minute,
            },
            'make-source-plan': {
                'specs': [{'bounds_x': [-0.5, 0.5], 'bounds_y': [-0.5, 0.5]}],
                'spec_shape': 'ellipse',
                'max_obs_length': 1200, # seconds
            },
            'make-source-scan': {
                'preferred_length': 1000,
            }
        },
        'calibration_targets': ["uranus"],
        'soft_targets': ["saturn"],
    }
    policy = policies.BasicPolicy(**config)
    seqs = policy.init_seqs(
        dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 10, 1, 0, 0, tzinfo=dt.timezone.utc))
    policy.apply(seqs)

