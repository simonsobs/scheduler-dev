import numpy as np
from dataclasses import dataclass
import datetime as dt

from .. import source as src, utils as u
from .sat import SchedMode, SATPolicy, State, CalTarget

logger = u.init_logger(__name__)


# ----------------------------------------------------
#
#         setup satp1 specific configs
#
# ----------------------------------------------------

def make_geometry():
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

def make_cal_target(
    source: str, 
    boresight: int, 
    elevation: int, 
    focus: str, 
    allow_partial=False,
    drift=True,
) -> CalTarget:
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

    return CalTarget(
        source=source, 
        array_query=array_focus[boresight][focus], 
        el_bore=elevation, 
        boresight_rot=boresight, 
        tag=tags[focus],
        allow_partial=allow_partial,
        drift=drift
    )

def make_blocks(master_file):
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

def make_operations(az_speed, az_accel, disable_hwp=False, apply_boresight_rot=True, hwp_cfg=None):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper' }
    pre_session_ops = [
        { 'name': 'sat.preamble'        , 'sched_mode': SchedMode.PreSession, 'hwp_cfg': hwp_cfg, },
        { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'sat.set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreSession, 'disable_hwp': disable_hwp, },
    ]
    cal_ops = [
        { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, },
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, },
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'indent': 4},
    ]
    cmb_ops = [
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreObs, },
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, },
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs, },
        { 'name': 'sat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostObs, },
    ]
    post_session_ops = [
        { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, },
        { 'name': 'sat.wrap_up'         , 'sched_mode': SchedMode.PostSession, 'az_stow': 180, 'el_stow': 48},
    ]
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    az_speed,
    az_accel,
    cal_targets,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(az_speed, az_accel, **op_cfg)

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
        'operations': operations,
        'allow_partial': False,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'az_speed' : az_speed,
        'az_accel' : az_accel,
    }
    return config


# ----------------------------------------------------
#
#         Policy customizations, if any
#
# ----------------------------------------------------
# here we add some convenience wrappers

@dataclass
class SATP1Policy(SATPolicy):
    @classmethod
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5, cal_targets=[], **op_cfg):
        return cls(**make_config(master_file, az_speed, az_accel, cal_targets, **op_cfg))

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for satp1, if needed"""
        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=0,
            hwp_spinning=False,
        )