"""A production-level implementation of the SAT policy

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

logger = u.init_logger(__name__)

@dataclass(frozen=True)
class State(tel.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands` as well as fields:
    (`boresight_rot_now`, `last_ufm_relock`, `last_bias_step`, `is_det_setup`)
    from State defined in `schedlib.policies.tel`.

    Parameters
    ----------
    hwp_spinning : bool
        Whether the high-precision measurement wheel is spinning or not.
    """
    hwp_spinning: bool = False


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

@cmd.operation(name="sat.preamble", duration=0)
def preamble(hwp_cfg):
    base = tel.preamble()
    append = ["sup = OCSClient('hwp-supervisor')", "",]
    return base + append

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state):
    return tel.ufm_relock(state)

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, disable_hwp=False, forward=True):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif state.hwp_spinning:
        return state, 0, ["# hwp already spinning"]
    else:
        state = state.replace(hwp_spinning=True)
        if forward:
            freq = 2
        else:
            freq = -2
        return state, 20*u.minute, [
            "sup.enable_driver_board()",
            f"run.hwp.set_freq(freq={freq})",
        ]

@cmd.operation(name='sat.hwp_spin_down', return_duration=True)
def hwp_spin_down(state, disable_hwp=False):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif not state.hwp_spinning:
        return state, 0, ["# hwp already stopped"]
    else:
        state = state.replace(hwp_spinning=False)
        return state, 10*u.minute, [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

# per block operation: block will be passed in as parameter
@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, apply_boresight_rot, iv_cadence)

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    block = block.trim_left_to(state.curr_time)
    if block is None:
        return state, 0, ["# too late, don't scan"]
    if (
        block.az_speed != state.az_speed_now or 
        block.az_accel != state.az_accel_now
    ):
        commands = [
            f"run.acu.set_scan_params({block.az_speed}, {block.az_accel})"
        ]
        state = state.replace(
            az_speed_now=block.az_speed, 
            az_accel_now=block.az_accel
        )
    else:
        commands = []
    
    state = state.replace(az_now=block.az, el_now=block.alt)
    commands.extend([
        "now = datetime.datetime.now(tz=UTC)",
        f"scan_start = {repr(block.t0)}",
        f"scan_stop = {repr(block.t1)}",
        f"if now > scan_start:",
        "    # adjust scan parameters",
        f"    az = {round(block.az,3)} + {round(block.az_drift,5)}*(now-scan_start).total_seconds()",
        f"else: ",
        f"    az = {round(block.az,3)}",
        f"if now > scan_stop:",
        "    # too late, don't scan",
        "    pass",
        "else:",
        f"    run.acu.move_to(az, {round(block.alt,3)})",
        "",
        f"    print('Waiting until {block.t0} to start scan')",
        f"    run.wait_until('{block.t0.isoformat()}')",
        "",
        "    run.seq.scan(",
        f"        description='{block.name}', ",
        f"        stop_time='{block.t1.isoformat()}', ",
        f"        width={round(block.throw,3)}, ",
        f"        az_drift={round(block.az_drift,5)}, ",
        f"        subtype='{block.subtype}',",
        f"        tag='{block.tag}',",
        "    )",
    ])
    return state, block.duration.total_seconds(), commands

@cmd.operation(name='sat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    return tel.setup_boresight(state, block, apply_boresight_rot)

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, min_interval=10*u.minute):
    return tel.bias_step(state, min_interval)

@cmd.operation(name='sat.wrap_up', duration=0)
def wrap_up(state, az_stow, el_stow):
    return tel.wrap_up(state, az_stow, el_stow)

@dataclass
class SATPolicy(tel.TelPolicy):
    """a more realistic SAT policy.

    Currently identical to `schedlib.policies.tel.TelPolicy` except for `init_state`.
    """
    def init_state(self, t0: dt.datetime) -> State:
        """
        Initializes the observatory state with some reasonable guess.
        In practice it should ideally be replaced with actual data
        from the observatory controller.

        Parameters
        ----------
        t0 : float
            The initial time for the state, typically representing the current time in a specific format.

        Returns
        -------
        State
        """
        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=0,
            hwp_spinning=False,
        )

# ------------------------
# utilities
# ------------------------

def simplify_hwp(op_seq):
    # if hwp is spinning up and down right next to each other, we can just remove them
    core.seq_assert_sorted(op_seq)
    def rewriter(seq_prev, b_next):
        if len(seq_prev) == 0:
            return [b_next]
        b_prev = seq_prev[-1]
        if (b_prev.name == 'sat.hwp_spin_up' and b_next.name == 'sat.hwp_spin_down') or \
           (b_prev.name == 'sat.hwp_spin_down' and b_next.name == 'sat.hwp_spin_up'):
            return seq_prev[:-1] + [cmd.OperationBlock(
                name='wait-until', 
                t0=b_prev.t0, 
                t1=b_next.t1, 
            )]
        else:
            return seq_prev+[b_next]
    return reduce(rewriter, op_seq, [])
