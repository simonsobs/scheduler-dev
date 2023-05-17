import os.path as op
import datetime as dt

minute = 60
hour = 60 * minute

config = {
    'master_schedule': op.dirname(__file__) + '/schedule_sat.txt',
    'rules': {
        'rephase-first': {
            'max_fraction': 0.1,
            'min_block_size': 600,  # in seconds
        },
        'sun-avoidance': {
            'min_angle_az': 6,
            'min_angle_alt': 6,
            'time_step': 30,   # sec
            'buffer_step': 10, # 30 * 10 = 5 mins
        },
        'day-mod': {
            'day': 0,
            'day_mod': 1,
            'day_ref': dt.datetime(2014, 1, 1, 0, 0, 0),
        },
        'min-duration': {
            'min_duration': 3 * minute,
        },
    },
    'calibration_targets': ["uranus"],
    'soft_targets': ["saturn"]
}
