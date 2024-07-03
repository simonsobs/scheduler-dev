"""A production-level implementation of the LAT policy

"""

import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field
import datetime as dt
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import reduce

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from . import tel
from .tel import State
from ..commands import SchedMode

logger = u.init_logger(__name__)


# ----------------------------------------------------
#                  Register operations
# ----------------------------------------------------
# Note: to avoid naming collisions. Use appropriate prefixes
# whenver necessary. For example, all satp1 specific
# operations should start with `satp1`.
#
# Registered operations can be three kinds of functions:
#
# 1. for operations with static duration, it can be defined as a function
#    that returns a list of commands, with the static duration specified in
#    the decorator
# 2. for operations with dynamic duration, meaning the duration is determined
#    at runtime, it can be defined as a function that returns a tuple of
#    duration and commands; the decorator should be informed with the option
#    `return_duration=True`
# 3. for operations that depends and/or modifies the state, the operation
#    function should take the state as the first argument (no renaming allowed)
#    and return a new state before the rest of the return values
#
# For example the following are all valid definitions:
#  @cmd.operation(name='my-op', duration=10)
#  def my_op():
#      return ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op():
#      return 10, ["do something"]
#
#  @cmd.operation(name='my-op')
#  def my_op(state):
#      return state, ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op(state):
#      return state, 10, ["do something"]


@cmd.operation(name="lat.preamble", duration=0)
def preamble():
    return tel.preamble()


@cmd.operation(name="lat.ufm_relock", return_duration=True)
def ufm_relock(state):
    return tel.ufm_relock(state)


# per block operation: block will be passed in as parameter
@cmd.operation(name="lat.det_setup", return_duration=True)
def det_setup(state, block, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, apply_boresight_rot, iv_cadence)


@cmd.operation(name="lat.cmb_scan", return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)


@cmd.operation(name="lat.source_scan", return_duration=True)
def source_scan(state, block):
    raise NotImplementedError("LAT source scans are not yet implemented")


@cmd.operation(name="lat.setup_boresight", return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    return tel.setup_boresight(state, block, apply_boresight_rot)


# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name="lat.bias_step", return_duration=True)
def bias_step(state, min_interval=10 * u.minute):
    return tel.bias_step(state, min_interval)


@cmd.operation(name="lat.wrap_up", duration=0)
def wrap_up(state, az_stow, el_stow):
    return tel.wrap_up(state, az_stow, el_stow)


@dataclass
class LATPolicy(tel.TelPolicy):
    """a more realistic LAT policy.

    Currently identical to `schedlib.policies.tel.TelPolicy` except with the addtion of `from_defaults`.
    """

    @classmethod
    def from_defaults(
        cls,
        master_file,
        az_speed=1,
        az_accel=1,
        cal_targets=[],
        iso_scan_speeds=None,
        boresight_override=None,
        **op_cfg
    ):
        return cls(
            **make_config(
                master_file,
                az_speed,
                az_accel,
                cal_targets,
                iso_scan_speeds,
                boresight_override,
                **op_cfg
            )
        )


# ----------------------------------------------------
#         setup LAT specific configs
# ----------------------------------------------------


def make_geometry():
    # These are just the median of the wafers and an ~overestimated rad rn
    # To be updated later
    return {
        "i1_ws0": {
            "center": [1.3516076803207397, 0.5679303407669067],
            "radius": 0.03,
        },
        "i1_ws1": {
            "center": [1.363024353981018, 1.2206860780715942],
            "radius": 0.03,
        },
        "i1_ws2": {
            "center": [1.9164373874664307, 0.9008757472038269],
            "radius": 0.03,
        },
        "i6_ws0": {
            "center": [1.3571038246154785, -1.2071731090545654],
            "radius": 0.03,
        },
        "i6_ws1": {
            "center": [1.3628365993499756, -0.5654135942459106],
            "radius": 0.03,
        },
        "i6_ws2": {
            "center": [1.9065929651260376, -0.8826764822006226],
            "radius": 0.03,
        },
    }


def make_cal_target(
    source: str,
    boresight: float,
    elevation: float,
    focus: str,
    allow_partial=False,
    drift=True,
) -> tel.CalTarget:
    raise NotImplementedError("Calibration targets not yet implemented for the LAT")


def make_blocks(master_file):
    return {
        "baseline": {"cmb": {"type": "toast", "file": master_file}},
        "calibration": {
            "saturn": {
                "type": "source",
                "name": "saturn",
            },
            "jupiter": {
                "type": "source",
                "name": "jupiter",
            },
            "moon": {
                "type": "source",
                "name": "moon",
            },
            "uranus": {
                "type": "source",
                "name": "uranus",
            },
            "neptune": {
                "type": "source",
                "name": "neptune",
            },
            "mercury": {
                "type": "source",
                "name": "mercury",
            },
            "venus": {
                "type": "source",
                "name": "venus",
            },
            "mars": {
                "type": "source",
                "name": "mars",
            },
        },
    }


def make_operations(
    az_speed, az_accel, apply_boresight_rot=True, iv_cadence=4 * u.hour
):
    pre_session_ops = [
        {
            "name": "lat.preamble",
            "sched_mode": SchedMode.PreSession,
        },
        {"name": "start_time", "sched_mode": SchedMode.PreSession},
        {
            "name": "lat.ufm_relock",
            "sched_mode": SchedMode.PreSession,
        },
        {
            "name": "set_scan_params",
            "sched_mode": SchedMode.PreSession,
            "az_speed": az_speed,
            "az_accel": az_accel,
        },
    ]
    cal_ops = [
        {
            "name": "lat.setup_boresight",
            "sched_mode": SchedMode.PreCal,
            "apply_boresight_rot": apply_boresight_rot,
        },
        {
            "name": "lat.det_setup",
            "sched_mode": SchedMode.PreCal,
            "apply_boresight_rot": apply_boresight_rot,
            "iv_cadence": iv_cadence,
        },
        {
            "name": "lat.source_scan",
            "sched_mode": SchedMode.InCal,
        },
        {"name": "lat.bias_step", "sched_mode": SchedMode.PostCal, "indent": 4},
    ]
    cmb_ops = [
        {
            "name": "lat.setup_boresight",
            "sched_mode": SchedMode.PreObs,
            "apply_boresight_rot": apply_boresight_rot,
        },
        {
            "name": "lat.det_setup",
            "sched_mode": SchedMode.PreObs,
            "apply_boresight_rot": apply_boresight_rot,
            "iv_cadence": iv_cadence,
        },
        {
            "name": "lat.bias_step",
            "sched_mode": SchedMode.PreObs,
        },
        {
            "name": "lat.cmb_scan",
            "sched_mode": SchedMode.InObs,
        },
        {
            "name": "lat.bias_step",
            "sched_mode": SchedMode.PostObs,
            "indent": 4,
            "divider": [""],
        },
    ]
    post_session_ops = [
        {
            "name": "sat.wrap_up",
            "sched_mode": SchedMode.PostSession,
            "az_stow": 180,
            "el_stow": 50,
        },
    ]
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops


def make_config(
    master_file,
    az_speed,
    az_accel,
    cal_targets,
    iso_scan_speeds=None,
    boresight_override=None,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(az_speed, az_accel, **op_cfg)

    sun_policy = {"min_angle": 41, "min_sun_time": 1980}

    config = {
        "blocks": blocks,
        "geometries": geometries,
        "rules": {
            "min-duration": {"min_duration": 600},
            "sun-avoidance": sun_policy,
        },
        "operations": operations,
        "cal_targets": cal_targets,
        "scan_tag": None,
        "iso_scan_speeds": iso_scan_speeds,
        "boresight_override": boresight_override,
        "az_speed": az_speed,
        "az_accel": az_accel,
        "stages": {
            "build_op": {
                "plan_moves": {
                    "sun_policy": sun_policy,
                    "az_step": 0.5,
                    "az_limits": [-45, 405],
                }
            }
        },
    }
    return config
