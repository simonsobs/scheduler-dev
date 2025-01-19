import numpy as np
from dataclasses import dataclass
import datetime as dt

from .. import source as src, utils as u
from .sat import SATPolicy, State, CalTarget
from ..commands import SchedMode

logger = u.init_logger(__name__)


# ----------------------------------------------------
#         setup satp3 specific configs
# ----------------------------------------------------

def make_geometry():
    ufm_mv12_shift = np.degrees([0, 0])
    ufm_mv35_shift = np.degrees([0, 0])
    ufm_mv23_shift = np.degrees([0, 0])
    ufm_mv5_shift  = np.degrees([0, 0])
    ufm_mv27_shift = np.degrees([0, 0])
    ufm_mv33_shift = np.degrees([0, 0])
    ufm_mv17_shift = np.degrees([0, 0])

    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    return {
      'ws3': {
        'center': [-d_xi+ufm_mv12_shift[0], d_eta_side+ufm_mv12_shift[1]],
        'radius': 6,
      },
      'ws2': {
        'center': [-d_xi+ufm_mv35_shift[0], -d_eta_side+ufm_mv35_shift[1]],
        'radius': 6,
      },
      'ws4': {
        'center': [0+ufm_mv23_shift[0], d_eta_mid+ufm_mv23_shift[1]],
        'radius': 6,
      },
      'ws0': {
        'center': [0+ufm_mv5_shift[0], 0+ufm_mv5_shift[1]],
        'radius': 6,
      },
      'ws1': {
        'center': [0+ufm_mv27_shift[0], -d_eta_mid+ufm_mv27_shift[1]],
        'radius': 6,
      },
      'ws5': {
        'center': [d_xi+ufm_mv33_shift[0], d_eta_side+ufm_mv33_shift[1]],
        'radius': 6,
      },
      'ws6': {
        'center': [d_xi+ufm_mv17_shift[0], -d_eta_side+ufm_mv17_shift[1]],
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
    az_branch=None,
    az_speed=None,
    az_accel=None,
) -> CalTarget:
    array_focus = {
        'left' : 'ws3,ws2',
        'middle' : 'ws0,ws1,ws4',
        'right' : 'ws5,ws6',
        'top': 'ws3,ws4,ws5',
        'toptop': 'ws4',
        'center': 'ws0',
        'bottom': 'ws1,ws2,ws6',
        'bottombottom': 'ws1',
        'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
    }

    boresight = int(boresight)
    elevation = int(elevation)
    focus = focus.lower()

    focus_str = None
    focus_str = array_focus.get(focus, focus)

    assert source in src.SOURCES, f"source should be one of {src.SOURCES.keys()}"

    if az_branch is None:
        az_branch = 180.

    return CalTarget(
        source=source,
        array_query=focus_str,
        el_bore=elevation,
        boresight_rot=boresight,
        tag=focus_str,
        allow_partial=allow_partial,
        drift=drift,
        az_branch=az_branch,
        az_speed=az_speed,
        az_accel=az_accel,
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
            },
            'rcw38': {
                'type' : 'source',
                'name' : 'rcw38',
            },
            'taua': {
                'type' : 'source',
                'name' : 'taua',
            },
            'galcenter': {
                'type' : 'source',
                'name' : 'galcenter',
            },
        },
    }

commands_uxm_relock = [
    "############# Daily Relock",
    "run.smurf.zero_biases()",
    "",
    "time.sleep(120)",
    "run.smurf.take_noise(concurrent=True, tag='res_check')",
    "run.smurf.uxm_relock(concurrent=True)",
    "run.smurf.take_bgmap(concurrent=True)",
    "",
]


commands_det_setup = [
    "",
    "################### Detector Setup######################",
    "with disable_trace():",
    "    run.initialize()",
    "pysmurfs = run.CLIENTS['smurf']",
    "run.smurf.take_bgmap(concurrent=True)",
    "run.smurf.iv_curve(concurrent=True)",
    "for smurf in pysmurfs:",
    "    smurf.bias_dets.start(rfrac=0.5, kwargs=dict(bias_groups=[0,1,2,3,4,5,6,7,8,9,10,11]))",
    "time.sleep(300)",
    "run.smurf.bias_step(concurrent=True)",
    "#################### Detector Setup Over ####################",
    "",
]

def make_operations(
    az_speed, az_accel, iv_cadence=4*u.hour, bias_step_cadence=0.5*u.hour,
    disable_hwp=False, apply_boresight_rot=False, hwp_cfg=None,
    home_at_end=False, run_relock=False,
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper'}

    pre_session_ops = [
        { 'name': 'sat.preamble'        , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'start_time'          ,'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]
    if run_relock:
        pre_session_ops += [
            { 'name': 'sat.ufm_relock'  , 'sched_mode': SchedMode.PreSession, 'commands': commands_uxm_relock, }
        ]
    cal_ops = [
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'commands': commands_det_setup, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops = [
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'commands': commands_det_setup, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence},
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs, 'bias_step_cadence': bias_step_cadence},
        { 'name': 'sat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
    ]
    if home_at_end:
        post_session_ops = [
            { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, },
            { 'name': 'end_time'          ,'sched_mode': SchedMode.PostSession},
        ]
    else:
        post_session_ops = []

    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    az_speed,
    az_accel,
    iv_cadence,
    bias_step_cadence,
    min_hwp_el,
    max_cmb_scan_duration,
    cal_targets,
    az_stow=None,
    el_stow=None,
    boresight_override=None,
    hwp_override=None,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(
        az_speed, az_accel,
        iv_cadence, bias_step_cadence,
        **op_cfg
    )

    if boresight_override is not None:
        logger.warning("Boresight Override does nothing for SATp3")
        boresight_override = None

    sun_policy = {
        'min_angle': 49,
        'min_sun_time': 1980,
        'min_el': 40,
    }

    if az_stow is None or el_stow is None:
        stow_position = {}
    else:
        stow_position = {
            'az_stow': az_stow,
            'el_stow': el_stow,
        }

    az_range = {
        'trim': False,
        'az_range': [-45, 405]
    }

    config = {
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'min-duration': {
                'min_duration': 600
            },
            'sun-avoidance': sun_policy,
            'az-range': az_range,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'hwp_override': hwp_override,
        'az_speed': az_speed,
        'az_accel': az_accel,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'min_hwp_el': min_hwp_el,
        'max_cmb_scan_duration': max_cmb_scan_duration,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'stow_position': stow_position,
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': az_range['az_range'],
                }
            }
        }
    }
    return config


# ----------------------------------------------------
#
#         Policy customizations, if any
#
# ----------------------------------------------------
# here we add some convenience wrappers

@dataclass
class SATP3Policy(SATPolicy):
    @classmethod
    def from_defaults(cls, master_file, az_speed=0.5, az_accel=0.25,
        iv_cadence=4*u.hour, bias_step_cadence=0.5*u.hour,
        min_hwp_el=48, max_cmb_scan_duration=1*u.hour,
        cal_targets=None, az_stow=None, el_stow=None,
        boresight_override=None, hwp_override=None,
        state_file=None, **op_cfg
    ):
        if cal_targets is None:
            cal_targets = []

        x = cls(**make_config(
            master_file, az_speed, az_accel,
            iv_cadence, bias_step_cadence, min_hwp_el,
            max_cmb_scan_duration, cal_targets,
            az_stow, el_stow, boresight_override,
            hwp_override, **op_cfg)
        )
        x.state_file = state_file
        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime, t1: dt.datetime) -> State:
        """customize typical initial state for satp1, if needed"""
        return State(
            curr_time=t0,
            end_time=t1,
            az_now=180,
            el_now=40,
            boresight_rot_now=0,
            hwp_spinning=False,
        )
