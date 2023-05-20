import os.path as op
import datetime as dt

minute = 60
hour = 60 * minute

config = {
    'master_schedule': op.dirname(__file__) + '/schedule_sat.txt',
    'rules': {
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
            'min_duration': 5 * minute,
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
    'calibration_targets': ["uranus"],
    'soft_targets': ["saturn"]
}
