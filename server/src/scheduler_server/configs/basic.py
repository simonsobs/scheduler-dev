import os.path as op
import datetime as dt

minute = 60
hour = 60 * minute

config = {
    'master_schedule': op.dirname(__file__) + '/schedule_sat.txt',
    'rules': {
        'sun-avoidance': {
            'min_angle_az': 6,  # deg
            'min_angle_alt': 6, # deg
            'time_step': 10,    # sec
            'n_buffer': 3,     # 30 * 1 = 0.5 mins
        },
        'day-mod': {
            'day': 0,
            'day_mod': 1,
            'day_ref': dt.datetime(2014, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        },
        'calibration-min-duration': {
            'min_duration': 5 * minute,
        },
        'alt-range': {
            'alt_range': [20, 90]
        },
        'make-source-plan': {
            'specs': [{'bounds_x': [-0.5, 0.5], 'bounds_y': [-0.5, 0.5]}],  # test
            'spec_shape': 'ellipse',
            'max_obs_length': 6000,
        },
        'make-source-scan': {
            'preferred_length': 1800,
        }
    },
    'calibration_targets': ["uranus", "saturn"],
    'soft_targets': []
}
