import numpy as np
from dataclasses import dataclass

from .. import source as src
from . import sat


def get_geometry():
    ufm_mv19_shift = np.degrees([-0.01583734, 0.00073145])
    ufm_mv15_shift = np.degrees([-0.01687046, -0.00117139])
    ufm_mv7_shift = np.degrees([-1.7275653e-02, -2.0664736e-06])
    ufm_mv9_shift = np.degrees([-0.01418133,  0.00820128])
    ufm_mv18_shift = np.degrees([-0.01625605,  0.00198077])
    ufm_mv22_shift = np.degrees([-0.0186627,  -0.00299793])
    ufm_mv29_shift = np.degrees([-0.01480562,  0.00117084])

    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    return {
        'ws3': {
            'center': [-d_xi+ufm_mv29_shift[0], d_eta_side+ufm_mv29_shift[1]],
            'radius': 6,
        },
        'ws2': {
            'center': [-d_xi+ufm_mv22_shift[0], -d_eta_side+ufm_mv22_shift[1]],
            'radius': 6,
        },
        'ws4': {
            'center': [0+ufm_mv7_shift[0], d_eta_mid+ufm_mv7_shift[1]],
            'radius': 6,
        },
        'ws0': {
            'center': [0+ufm_mv19_shift[0], 0+ufm_mv19_shift[1]],
            'radius': 6,
        },
        'ws1': {
            'center': [0+ufm_mv18_shift[0], -d_eta_mid+ufm_mv18_shift[1]],
            'radius': 6,
        },
        'ws5': {
            'center': [d_xi+ufm_mv9_shift[0], d_eta_side+ufm_mv9_shift[1]],
            'radius': 6,
        },
        'ws6': {
            'center': [d_xi+ufm_mv15_shift[0], -d_eta_side+ufm_mv15_shift[1]],
            'radius': 6,
        },
    }

def get_cal_target(source: str, boresight: int, elevation: int, focus: str):
    array_focus = {
        0 : {
            'left' : 'ws3,ws2',
            'middle' : 'ws0,ws1,ws4',
            'right' : 'ws5,ws6',
            'bottom': 'ws1,ws2,ws6',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
        45 : {
            'left' : 'ws3,ws4',
            'middle' : 'ws2,ws0,ws5',
            'right' : 'ws1,ws6',
            'bottom': 'ws1,ws2,ws3',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
        -45 : {
            'left' : 'ws1,ws2',
            'middle' : 'ws6,ws0,ws3',
            'right' : 'ws4,ws5',
            'bottom': 'ws1,ws6,ws5',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
    }
    tags = {
        'left': 'left_focal_plane',
        'middle': 'mid_focal_plane',
        'right': 'right_focal_plane',
        'bottom': 'bottom_focal_plane',
        'all': 'whole_focal_plane',
    }

    boresight = int(boresight)
    elevation = int(elevation)
    focus = focus.lower()

    assert boresight in array_focus, f"boresight should be one of {array_focus.keys()}"
    assert focus in tags, f"array_focus should be one of {tags.keys()}"
    assert source in src.SOURCES, f"source should be one of {src.SOURCES.keys()}"

    return (source, array_focus[boresight][focus], elevation, boresight, tags[focus])

def get_blocks(master_file):
    return {
        'baseline': {
            'cmb': {
                'type': 'toast',
                'file': master_file
            }
        },
        'calibration': {
            'saturn': {
                'type' : 'source',
                'name' : 'saturn',
            },
            'jupiter': {
                'type' : 'source',
                'name' : 'jupiter',
            },
            'moon': {
                'type' : 'source',
                'name' : 'moon',
            },
            'uranus': {
                'type' : 'source',
                'name' : 'uranus',
            },
            'neptune': {
                'type' : 'source',
                'name' : 'neptune',
            },
            'mercury': {
                'type' : 'source',
                'name' : 'mercury',
            },
            'venus': {
                'type' : 'source',
                'name' : 'venus',
            },
            'mars': {
                'type' : 'source',
                'name' : 'mars',
            }
        },
    }

def get_config(master_file, az_speed, az_accel, cal_targets):
    blocks = get_blocks(master_file)
    geometries = get_geometry()

    config = {
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'sun-avoidance': {
                'min_angle': 45,
            },
            'min-duration': {
                'min_duration': 600
            },
        },
        'allow_partial': False,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'az_speed' : az_speed,
        'az_accel' : az_accel,
    }
    return config


@dataclass
class SATP1Policy(sat.SATPolicy):

    @classmethod
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5, cal_targets=[]):
        return cls(**get_config(master_file, az_speed, az_accel, cal_targets))

    def add_cal_target(self, source: str, boresight: int, elevation: int, focus: str):
        self.cal_targets.append(get_cal_target(source, boresight, elevation, focus))

