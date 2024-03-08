"""A production-level implementation of the SAT policy

"""
import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field
import datetime as dt
from typing import List, Union, Optional, Dict, Any
import jax.tree_util as tu
from functools import reduce

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance, get_sun_tracker

logger = u.init_logger(__name__)

class SchedMode:
    """
    Enumerate different options for scheduling operations in SATPolicy.

    Attributes
    ----------
    PreCal : str
        'pre_cal'; Operations scheduled before block.t0 for calibration.
    PreObs : str
        'pre_obs'; Observations scheduled before block.t0 for observation.
    InCal : str
        'in_cal'; Calibration operations scheduled between block.t0 and block.t1.
    InObs : str
        'in_obs'; Observation operations scheduled between block.t0 and block.t1.
    PostCal : str
        'post_cal'; Calibration operations scheduled after block.t1.
    PostObs : str
        'post_obs'; Observations operations scheduled after block.t1.
    PreSession : str
        'pre_session'; Represents the start of a session, scheduled from the beginning of the requested t0.
    PostSession : str
        'post_session'; Indicates the end of a session, scheduled after the last operation.

    """
    PreCal = 'pre_cal'
    PreObs = 'pre_obs'
    InCal = 'in_cal'
    InObs = 'in_obs'
    PostCal = 'post_cal'
    PostObs = 'post_obs'
    PreSession = 'pre_session'
    PostSession = 'post_session'


@dataclass(frozen=True)
class State(cmd.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.

    Parameters
    ----------
    boresight_rot_now : int
        The current boresight rotation state.
    hwp_spinning : bool
        Whether the high-precision measurement wheel is spinning or not.
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    """
    boresight_rot_now: int = 0
    hwp_spinning: bool = False
    last_ufm_relock: Optional[dt.datetime] = None
    last_bias_step: Optional[dt.datetime] = None
    is_det_setup: bool = False


@dataclass(frozen=True)
class CalTarget:
    source: str
    array_query: str
    el_bore: float
    tag: str
    boresight_rot: float = 0
    allow_partial: bool = False
    drift: bool = True

# ----------------------------------------------------
#
#                  Register operations
#
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
    return [
    "import time",
    "import datetime",
    "",
    "import numpy as np",
    "import sorunlib as run",
    "from ocs.ocs_client import OCSClient",
    "",
    "run.initialize()",
    "",
    "UTC = datetime.timezone.utc",
    "acu = run.CLIENTS['acu']",
    "pysmurfs = run.CLIENTS['smurf']",
    "",
    "# HWP Params",
    "use_pid = True",
    "forward = True",
    "hwp_freq = 2.0",
    "",
    "def HWPPrep():",
    f"    iboot2 = OCSClient('{hwp_cfg['iboot2']}')",
    "    iboot2.set_outlet(outlet = 1, state = 'on')",
    "    iboot2.set_outlet(outlet = 2, state = 'on')",
    "",
    f"    pid = OCSClient('{hwp_cfg['pid']}')",
    f"    pmx = OCSClient('{hwp_cfg['pmx']}')",
    "    pid.acq.stop()",
    "    global use_pid",
    "    global forward",
    "",
    "    if use_pid:",
    "        pmx.use_ext()",
    "    else:",
    "        pmx.ign_ext()",
    "",
    "    if forward:",
    "        pid.set_direction(direction = '0')",
    "    else:",
    "        pid.set_direction(direction = '1')",
    "    pid.acq.start()",
    "",
    "def HWPPost():",
    f"    iboot2 = OCSClient('{hwp_cfg['iboot2']}')",
    f"    gripper = OCSClient('{hwp_cfg['gripper']}')",
    "    iboot2.set_outlet(outlet = 1, state = 'off')",
    "    iboot2.set_outlet(outlet = 2, state = 'off')",
    "    gripper.force(value = False)",
    "    gripper.brake(state = True)",
    "    gripper.power(state = False)",
    "",
    "def HWPSpinUp():",
    f"    pid = OCSClient('{hwp_cfg['pid']}')",
    f"    pmx = OCSClient('{hwp_cfg['pmx']}')",
    "    pid.acq.stop()",
    "    global use_pid",
    "    global forward",
    "    global hwp_freq",
    "",
    "    if use_pid:",
    "        if forward:",
    "            pid.set_direction(direction = '0')",
    "        else:",
    "            pid.set_direction(direction = '1')",
    "",
    "        pid.declare_freq(freq = hwp_freq)",
    "        pid.tune_freq()",
    "        pmx.set_on()",
    "",
    "        time.sleep(1)",
    "        cur_freq = float(pid.acq.status().session['data']['current_freq'])",
    "",
    "        while abs(cur_freq - hwp_freq) > 0.005:",
    "            cur_freq = float(pid.acq.status().session['data']['current_freq'])",
    "            print ('Current Frequency =', cur_freq, 'Hz    ', end = '\\r')",
    "",
    "        print('                                    ', end = '\\r')",
    "        print('Tuning finished')",
    "    else:",
    "        print('Error: Not using PID')",
    "",
    "    pid.acq.start()",
    "",
    "def HWPFastStop():",
    f"    iboot2 = OCSClient('{hwp_cfg['iboot2']}')",
    f"    pid = OCSClient('{hwp_cfg['pid']}')",
    f"    pmx = OCSClient('{hwp_cfg['pmx']}')",
    "    pid.acq.stop()",
    "    global use_pid",
    "    global forward",
    "",
    "    if use_pid:",
    "        print('Starting stop')",
    "        if forward:",
    "            pid.set_direction(direction = '1')",
    "        else:",
    "            pid.set_direction(direction = '0')",
    "",
    "        pid.tune_stop()",
    "        pmx.set_on()",
    "",
    "        time.sleep(1)",
    "        start_freq = float(pid.get_freq()[2]['messages'][1][1].split(' ')[3])",
    "        time.sleep(15)",
    "        cur_freq = float(pid.get_freq()[2]['messages'][1][1].split(' ')[3])",
    "        if cur_freq > start_freq:",
    "            if forward:",
    "                pid.set_direction(direction = '0')",
    "            else:",
    "                pid.set_direction(direction = '1')",
    "",
    "            start_freq = cur_freq",
    "            time.sleep(15)",
    "            cur_freq = float(pid.get_freq()[2]['messages'][1][1].split(' ')[3])",
    "            if cur_freq > start_freq:",
    "                pmx.set_off()",
    "                iboot2.set_outlet(outlet = 1, state = 'off')",
    "                iboot2.set_outlet(outlet = 2, state = 'off')",
    "                time.sleep(60*30)",
    "",
    "        while cur_freq > 0.2:",
    "            cur_freq = float(pid.get_freq()[2]['messages'][1][1].split(' ')[3])",
    "            print ('Current Frequency =', cur_freq, 'Hz    ', end = '\\r')",
    "",
    "        pmx.set_off()",
    "        iboot2.set_outlet(outlet = 1, state = 'off')",
    "        iboot2.set_outlet(outlet = 2, state = 'off')",
    "        time.sleep(180)",
    "        iboot2.set_outlet(outlet = 1, state = 'on')",
    "        iboot2.set_outlet(outlet = 2, state = 'on')",
    "",
    "        print('                                    ', end = '\\r')",
    "        print('CHWP stopped')",
    "    else:",
    "        print('Error: Not using PID')",
    "",
    "    pid.acq.start()",
    "",
    ]

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state):
    if state.last_ufm_relock is None:
        doit = True
    elif (state.curr_time - state.last_ufm_relock).total_seconds() > 12*u.hour:
        doit = True
    else:
        doit = False

    if doit:
        state = state.replace(last_ufm_relock=state.curr_time)
        return state, 15*u.minute, [
            "for smurf in pysmurfs:",
            "    smurf.zero_biases.start()",
            "for smurf in pysmurfs:",
            "    smurf.zero_biases.wait()",
            "",
            "time.sleep(120)",
            "run.smurf.take_noise(concurrent=True, tag='oper,take_noise,res_check')",
            "run.smurf.uxm_relock(concurrent=True)",
            "",
        ]
    else:
        return state, 0, ["# no ufm relock needed at this time"]

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, disable_hwp=False):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif state.hwp_spinning:
        return state, 0, ["# hwp already spinning"]
    else:
        state = state.replace(hwp_spinning=True)
        return state, 20*u.minute, [
            "HWPPrep()",
            "forward = True",
            "hwp_freq = 2.0",
            "HWPSpinUp()",
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
            "HWPFastStop()",
            "HWPPost()",
            "hwp_freq = 0.0",
        ]

# per block operation: block will be passed in as parameter
@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block):
    # when should det setup be done?
    # -> should always be done if the block is a cal block
    # -> should always be done if elevation has changed
    # -> should always be done if det setup has not been done yet
    # -> should always be done if boresight rotation has changed
    doit = (block.subtype == 'cal') or (block.alt != state.el_now) or \
        (not state.is_det_setup) or (block.boresight_angle != state.boresight_rot_now)

    if doit:
        commands = [
            "",
            "################### Detector Setup######################",
            "run.smurf.take_bgmap(concurrent=True)",
            "run.smurf.iv_curve(concurrent=False, settling_time=0.1)",
            "run.smurf.bias_dets(concurrent=True)",
            "time.sleep(180)",
            "run.smurf.bias_step(concurrent=True)",
            "#################### Detector Setup Over ####################",
            "",
        ]
        state = state.replace(
            last_bias_step=state.curr_time,
            is_det_setup=True
        )
        return state, 40*u.minute, commands
    else:
        return state, 0, []

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    commands = [
        "run.seq.scan(",
        f"    description='{block.name}',",
        f"    stop_time='{block.t1.isoformat()}',",
        f"    width={round(block.throw,3)}, az_drift=0,",
        f"    subtype='cmb', tag='{block.tag}',",
        ")",
    ]
    return state, (block.t1 - state.curr_time).total_seconds(), commands

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    block = block.trim_left_to(state.curr_time)
    if block is None:
        return 0, ["# too late, don't scan"]
    state = state.replace(az_now=block.az, el_now=block.alt)
    return state, block.duration.total_seconds(), [
        "now = datetime.datetime.now(tz=UTC)",
        f"scan_start = {repr(block.t0)}",
        f"scan_stop = {repr(block.t1)}",
        f"if now > scan_start:",
        "    # adjust scan parameters",
        f"    az = {round(np.mod(block.az,360),3)} + {round(block.az_drift,5)}*(now-scan_start).total_seconds()",
        f"else: ",
        f"    az = {round(np.mod(block.az,360),3)}",
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
    ]

@cmd.operation(name='sat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0

    if apply_boresight_rot and state.boresight_rot_now != block.boresight_angle:
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += 1*u.minute

    # hard to know which az we stopped at, always move
    state, dur, cmds = cmd.make_op('move_to', block.az, block.alt)(state)
    commands += cmds
    duration += dur

    return state, duration, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, min_interval=10*u.minute):
    if state.last_bias_step is None or (state.curr_time - state.last_bias_step).total_seconds() > min_interval:
        state = state.replace(last_bias_step=state.curr_time)
        return state, 60, [ "run.smurf.bias_step(concurrent=True)" ]
    else:
        return state, 0, []

@cmd.operation(name='sat.wrap_up', duration=0)
def wrap_up(state, az_stow, el_stow):
    state = state.replace(az_now=az_stow, el_now=el_stow)
    return state, [
        "# go home",
        f"run.acu.move_to(az={az_stow}, el={el_stow})",
        "time.sleep(1)"
    ]

@cmd.operation(name='sat.set_scan_params', duration=0)
def set_scan_params(state, az_speed, az_accel):
    if az_speed != state.az_speed_now or az_accel != state.az_accel_now:
        state = state.replace(az_speed_now=az_speed, az_accel_now=az_accel)
        return state, [ f"run.acu.set_scan_params({az_speed}, {az_accel})"]
    return state, []

# some generic operations that will be moved out of the module soon
@cmd.operation(name='wait_until', return_duration=True)
def wait_until(state, t1: dt.datetime):
    return state, (t1-state.curr_time).total_seconds(), [
        f"run.wait_until('{t1.isoformat()}')"
    ]

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, n_points=100, min_angle=45, min_el=0, az_limit=[-90, 405]):
    duration = 60

    # find possible unwraps within the allowed az limits
    az = (az - az_limit[0]) % 360 + az_limit[0]  # az -> [-90, 270)
    az_options = np.arange(az, az_limit[-1], 360)

    # sort them based on their distance to current az
    idx = np.argsort(np.abs((az_options - az + 180) % 360 - 180))
    az_options = az_options[idx]

    sun_tracker = get_sun_tracker(u.dt2ct(state.curr_time))

    for az_option in az_options:
        _path = sun_tracker.analyze_paths(state.az_now, state.el_now, az_option, el, 
                                          dodging=False, t=u.dt2ct(state.curr_time))
        best_path = sun_tracker.select_move(_path)[0]
        if best_path is not None:
            break
         
    if best_path is None or (not best_path['direct']):
        logger.error(f"--> No safe direct moves found, tried all options: {az_options}")

    az, el = best_path['moves'].nodes[0]
    state = state.replace(az_now=az, el_now=el)

    return state, duration, [f"run.acu.move_to(az={round(az, 3)}, el={round(el, 3)})"]

@dataclass
class SATPolicy:
    """a more realistic SAT policy.

    Parameters
    ----------
    blocks : dict
        a dict of blocks, with keys 'baseline' and 'calibration'
    rules : dict
        a dict of rules, specifies rule cfgs for e.g., 'sun-avoidance', 'az-range', 'min-duration'
    geometries : dict
        a dict of geometries, with the leave node being dict with keys 'center' and 'radius'
    cal_targets : list[CalTarget]
        a list of calibration target each described by CalTarget object
    cal_policy : str
        calibration policy: default to round-robin
    scan_tag : str
        a tag to be added to all scans
    az_speed : float
        the az speed in deg / s
    az_accel : float
        the az acceleration in deg / s^2
    wafer_sets : dict[str, str]
        a dict of wafer sets definitions
    operations : List[Dict[str, Any]]
        an orderred list of operation configurations
    """
    blocks: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, core.Rule] = field(default_factory=dict)
    geometries: List[Dict[str, Any]] = field(default_factory=list)
    cal_targets: List[CalTarget] = field(default_factory=list)
    cal_policy: str = 'round-robin'
    scan_tag: Optional[str] = None
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s^2
    allow_az_maneuver: bool = True
    wafer_sets: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str]):
        """
        Constructs a policy object from a YAML configuration file, a YAML string, or a dictionary.

        Parameters
        ----------
        config : Union[dict, str]
            The configuration to populate the policy object.

        Returns
        -------
        The constructed policy object.
        """
        if isinstance(config, str):
            loader = cfg.get_loader()
            if op.isfile(config):
                with open(config, "r") as f:
                    config = yaml.load(f.read(), Loader=loader)
            else:
                config = yaml.load(config, Loader=loader)
        return cls(**config)

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        """
        Initialize the sequences for the scheduler to process.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.
        t1 : datetime.datetime
            The end time of the sequences.

        Returns
        -------
        BlocksTree (nested dict / list of blocks)
            The initialized sequences
        """
        def construct_seq(loader_cfg):
            if loader_cfg['type'] == 'source':
                return src.source_gen_seq(loader_cfg['name'], t0, t1)
            elif loader_cfg['type'] == 'toast':
                return inst.parse_sequence_from_toast(loader_cfg['file'])
            else:
                raise ValueError(f"unknown sequence type: {loader_cfg['type']}")

        # construct seqs by traversing the blocks definition dict
        blocks = tu.tree_map(construct_seq, self.blocks,
                             is_leaf=lambda x: isinstance(x, dict) and 'type' in x)

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            source = cal_target.source
            if source not in blocks['calibration']:
                blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)

        # update az speed in scan blocks
        blocks = core.seq_map_when(
            lambda b: isinstance(b, inst.ScanBlock),
            lambda b: b.replace(az_speed=self.az_speed),
            blocks
        )

        # trim to given time range
        blocks = core.seq_trim(blocks, t0, t1)

        # ok to drop Nones
        blocks = tu.tree_map(
            lambda x: [x_ for x_ in x if x_ is not None],
            blocks,
            is_leaf=lambda x: isinstance(x, list)
        )

        return blocks

    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        """
        Applies a set of observing rules to the a tree of blocks such as modifying
        it with sun avoidance constraints and planning source scans for calibration.

        Parameters
        ----------
        blocks : BlocksTree
            The original blocks tree structure defining observing sequences and constraints.

        Returns
        -------
        BlocksTree
            New blocks tree after applying the specified observing rules.

        """
        # -----------------------------------------------------------------
        # step 1: preliminary sun avoidance
        #   - get rid of source observing windows too close to the sun
        #   - likely won't affect scan blocks because master schedule already
        #     takes care of this
        # -----------------------------------------------------------------
        if 'sun-avoidance' in self.rules:
            logger.info(f"applying sun avoidance rule: {self.rules['sun-avoidance']}")
            sun_rule = SunAvoidance(**self.rules['sun-avoidance'])
            blocks = sun_rule(blocks)
        else:
            logger.error("no sun avoidance rule specified!")
            raise ValueError("Sun rule is required!")

        # -----------------------------------------------------------------
        # step 2: plan calibration scans
        #   - refer to each target specified in cal_targets
        #   - same source can be observed multiple times with different
        #     array configurations (i.e. using array_query)
        # -----------------------------------------------------------------
        logger.info("planning calibration scans...")
        cal_blocks = []

        for target in self.cal_targets:
            logger.info(f"-> planning calibration scans for {target}...")
            assert target.source in blocks['calibration'], f"source {target.source} not found in sequence"

            # digest array_query: it could be a fnmatch pattern matching the path
            # in the geometry dict, or it could be looked up from a predefined
            # wafer_set dict. Here we account for the latter case:
            # look up predefined query in wafer_set
            if target.array_query in self.wafer_sets:
                array_query = self.wafer_sets[target.array_query]
            else:
                array_query = target.array_query

            # build array geometry information based on the query
            array_info = inst.array_info_from_query(self.geometries, array_query)
            logger.debug(f"-> array_info: {array_info}")

            # apply MakeCESourceScan rule to transform known observing windows into
            # actual scans
            rule = ru.MakeCESourceScan(
                array_info=array_info,
                el_bore=target.el_bore,
                drift=True,
                boresight_rot=target.boresight_rot,
                allow_partial=target.allow_partial,
            )
            source_scans = rule(blocks['calibration'][target.source])
            source_scans = core.seq_flatten(source_scans)

            # sun check again: previous sun check ensure source is not too
            # close to the sun, but our scan may still get close enough to
            # the sun, in which case we will trim it or delete it depending
            # on whether allow_partial is True
            if target.allow_partial:
                logger.info("-> allow_partial = True: trimming scan options by sun rule")
                min_dur_rule = ru.make_rule('min-duration', **self.rules['min-duration'])
                source_scans = core.seq_flatten(min_dur_rule(sun_rule(source_scans)))
            else:
                logger.info("-> allow_partial = False: filtering scan options by sun rule")
                source_scans = core.seq_flatten(
                    core.seq_filter(lambda b: b == sun_rule(b), source_scans)
                )

            # add tags to the scans
            cal_blocks.append(core.seq_map(
                lambda block: block.replace(tag=f"{block.tag},{target.tag}"),
                source_scans
            ))

            logger.info(f"-> found {len(source_scans)} scan options for {target.source} ({target.array_query}): {u.pformat(source_scans)}")

        # -----------------------------------------------------------------
        # step 3: resolve calibration target conflicts
        #   currently we adopt a simple round-robin strategy to resolve
        #   conflicts between multiple calibration targets. This is done
        #   by cycling through the calibration targets and add scan blocks
        #   successively in the order given in the cal_targets config.
        # -----------------------------------------------------------------

        try:
            # currently only implemented round-robin approach, but can be extended to other strategies
            cal_policy = { 'round-robin': round_robin }[self.cal_policy]
        except KeyError:
            raise ValueError(f"unsupported calibration policy: {self.cal_policy}")

        # done with the calibration blocks
        logger.info(f"applying calibration policy - {self.cal_policy} - to resolve calibration target conflicts")
        blocks['calibration'] = core.seq_resolve_overlap(list(cal_policy(cal_blocks, sun_avoidance=sun_rule)))
        logger.info(f"-> after calibration policy: {u.pformat(blocks['calibration'])}")

        # check sun avoidance again
        blocks['calibration'] = core.seq_flatten(sun_rule(blocks['calibration']))

        # min duration rule
        if 'min-duration' in self.rules:
            logger.info(f"applying min duration rule: {self.rules['min-duration']}")
            rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            blocks = rule(blocks)

        # -----------------------------------------------------------------
        # step 4: tags
        # -----------------------------------------------------------------

        # add proper subtypes
        blocks['calibration'] = core.seq_map(
            lambda block: block.replace(subtype="cal"),
            blocks['calibration']
        )

        blocks['baseline']['cmb'] = core.seq_map(
            lambda block: block.replace(
                subtype="cmb",
                tag=f"{block.az:.0f}-{block.az+block.throw:.0f}"
            ),
            blocks['baseline']['cmb']
        )

        # add scan tag if supplied
        if self.scan_tag is not None:
            blocks['baseline'] = core.seq_map(
                lambda block: block.replace(tag=f"{block.tag},{self.scan_tag}"),
                blocks['baseline']
            )

        return blocks

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

    def seq2cmd(
        self,
        seq: core.Blocks,
        t0: dt.datetime,
        t1: dt.datetime,
        state: Optional[State] = None
    ) -> str:
        """
        Converts a sequence of blocks into a list of commands to be executed
        between two given times.

        This method is responsible for generating commands based on a given
        sequence of observing blocks, considering specific hardware settings and
        constraints. It also includes timing considerations, such as time to
        relock a UFM or boresight angles, and ensures proper settings for
        azimuth speed and acceleration. It is assumed that the provided sequence
        is sorted in time.

        Parameters
        ----------
        seq : core.Blocks
            A tree-like sequence of Blocks representing the observation schedule
        t0 : datetime.datetime
            The starting datetime for the command sequence.
        t1 : datetime.datetime
            The ending datetime for the command sequence
        state : Optional[State], optional
            The initial state of the observatory, by default None

        Returns
        -------
        list of OperationBlock

        """
        op_seq = []

        # create state if not provided
        if state is None:
            state = self.init_state(t0)
            logger.debug(f"initial state: {u.pformat(state)}")

        # -----------------------------------------------------------------
        # 1. pre-session operations
        # -----------------------------------------------------------------
        logger.info("---------- step 1: planning pre-session ops ----------")

        ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PreSession]
        state, _, block_ops = self._apply_ops(state, ops)
        op_seq += block_ops

        post_init_state = state
        post_init_op_seq = op_seq.copy()

        logger.debug(f"post-init op_seq: {u.pformat(op_seq)}")
        logger.info(f"post-init state: {state}")

        # -----------------------------------------------------------------
        # 2. calibration scans
        #
        #    Note: in full generality we shouldn't take calibration scans out of
        #    the context, because other non-cal operations might have altered
        #    the state of the observatory in between. Therefore, the fact that
        #    we are planning calibration scans separately is based on the assumption
        #    that other operations we will try to preserve the relevant state for
        #    calibration. For example, if hwp is spinning, other operations should
        #    try to get hwp back to spinning when the operations are done.
        #
        # -----------------------------------------------------------------
        logger.info("---------- step 2: planning calibration scans ----------")

        # compile the blocks to plan
        cal_blocks = core.seq_sort(seq['calibration'], flatten=True)
        cmb_blocks = core.seq_sort(seq['baseline']['cmb'], flatten=True)

        # compile calibration operations
        pre_ops  = [op for op in self.operations if op['sched_mode'] == SchedMode.PreCal]
        in_ops   = [op for op in self.operations if op['sched_mode'] == SchedMode.InCal]
        post_ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostCal]
        cal_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        # compile cmb operations (also needed for state propagation)
        pre_ops  = [op for op in self.operations if op['sched_mode'] == SchedMode.PreObs]
        in_ops   = [op for op in self.operations if op['sched_mode'] == SchedMode.InObs]
        post_ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostObs]
        cmb_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        logger.debug(f"cal_ops to plan: {cal_ops}")
        logger.debug(f"pre-cal state: {u.pformat(state)}")

        # calibration always take precedence, merging gets rid of overlapping regions
        all_blocks = core.seq_merge(cmb_blocks, cal_blocks, flatten=True)

        cal_ref_state = state

        # need to loop over all blocks to get the state right
        for block in all_blocks:
            # cmb scans are just used to get the state right, no operations
            # are actually added to our list
            if block.subtype == 'cmb':
                constraint = core.Block(t0=state.curr_time, t1=block.t1)
                # probably not worth doing sun avoidance just for getting state right
                state, _ = self._plan_block_operations(state, block, constraint, **cmb_ops)
            elif block.subtype == 'cal':
                logger.info(f"-> planning cal block: {u.pformat(block)}")
                logger.info(f"--> pre-cal state: {state}")

                # skip if we are already past the block to avoid computation of sun cover
                if state.curr_time >= block.t1:
                    logger.info(f"--> skipping cal block {block.name} because it's already past")
                    continue

                # calibration takes highest precedence, so it will only be constrained by sun safety
                # and internal constraints (not overlapping with other cal blocks), so we will keep
                # track of the state after each cal block with cal_ref_state and use that as the
                # hard starting point constraint
                constraint = self._get_sun_safe_interval_for_block(
                    block, state, t_start=cal_ref_state.curr_time)
                ops_ = cal_ops

                # To give calibration maximal priority, we will ignore the state curr_time
                # and assume we can start whenever we want, as long as we are sun safe and
                # we are done with previous calibration block. Here we dial our clock back
                # to the start of the sun safe interval.
                state = state.replace(curr_time=constraint.t0)

                if self.allow_az_maneuver:
                    ops_, constraint = self._revise_ops_constraint_with_maneuver(
                        ops_, constraint, block, state)

                state, block_ops = self._plan_block_operations(state, block, constraint, **ops_)

                logger.debug(f"--> post-block ops: {u.pformat(block_ops)}")
                logger.debug(f"--> post-block state: {u.pformat(state)}")

                # update our reference state
                cal_ref_state = state

                op_seq += block_ops
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        logger.debug(f"final state after cal planning: {u.pformat(state)}")

        # -----------------------------------------------------------------
        # 3. cmb scans
        #
        # Note: for cmb scans, we will avoid overlapping with calibrations;
        # this means we will tend to overwrite into cmb blocks more often
        # -----------------------------------------------------------------
        logger.info("---------- step 3: planning cmb ops ----------")

        # restart from post-init
        state = post_init_state

        # calibration operations take precedence: now we know how long these cal
        # operations will take, we can remove overlapping regions from our cmb scans

        # avoid pre-cmb operations from colliding with calibration operations,
        # so identify non-overlapping intervals here. This will be used as constraint
        full_interval = core.Block(t0=t0, t1=t1)
        non_overlaps = core.seq_remove_overlap(full_interval, op_seq)

        # also remove overlapping intervals from cmb_blocks
        cmb_blocks = core.seq_remove_overlap(cmb_blocks, op_seq)

        # merge all blocks
        all_blocks = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        if 'min-duration' in self.rules:
            logger.info(f"applying min duration rule: {self.rules['min-duration']}")
            rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            all_blocks = core.seq_flatten(rule(all_blocks))

        # done with previously planned operation seqs
        # re-plan from the end of initialization.
        state = post_init_state
        op_seq = post_init_op_seq

        for block in all_blocks:
            if state.curr_time >= block.t1:
                logger.debug(f"--> skipping block {block.name} because it's already past")
                continue

            logger.info(f"-> planning block ({block.subtype}): {block.name}: {block.t0} - {block.t1}")
            logger.debug(f"--> pre-block state: {u.pformat(state)}")

            constraint = core.Block(t0=state.curr_time, t1=block.t1)
            logger.info(f"--> causal constraint: {constraint.t0} to {constraint.t1}")
            sun_constraint = self._get_sun_safe_interval_for_block(block, state)
            logger.info(f"--> sun constraint: {sun_constraint.t0} to {sun_constraint.t1}")
            constraint = core.block_intersect(constraint, sun_constraint)

            if block.subtype == 'cmb':
                non_overlap = [b for b in non_overlaps if b.t0 <= block.t0 and block.t1 <= b.t1][0]
                logger.info(f"--> operational constraint: {non_overlap.t0} to {non_overlap.t1}")
                constraint = core.block_intersect(non_overlap, sun_constraint)
                ops_ = cmb_ops
            elif block.subtype == 'cal':
                constraint = sun_constraint
                ops_ = cal_ops

                if self.allow_az_maneuver:
                    ops_, constraint = self._revise_ops_constraint_with_maneuver(
                        ops_, constraint, block, state)

            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")
            logger.info(f"--> final constraint: {constraint.t0} to {constraint.t1}")

            state, block_ops = self._plan_block_operations(state, block, constraint, **ops_)
            logger.debug(f"--> post-block state: {u.pformat(state)}")

            op_seq += block_ops

        logger.debug(f"post-planning state: {state}")

        # -----------------------------------------------------------------
        # 4. post-session operations
        # -----------------------------------------------------------------
        logger.info("---------- step 4: planning post-session ops ----------")

        ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostSession]
        state, _, post_session_ops = self._apply_ops(state, ops)
        op_seq += post_session_ops

        logger.debug(f"post-session state: {state}")
        logger.debug(f"post-session ops: {u.pformat(post_session_ops)}")

        logger.info("---------- step 5: post-processing ----------")

        # make sure operations are sorted by time, and we want to do that safely
        # without disturbing the original order of operations especially when
        # we have lots of no duration operations that look like they start at
        # the same time
        op_seq = core.seq_sort(op_seq)

        logger.info("replacing gaps in the operation sequence with wait-until...")
        # whenver there is a gap, replace it with a wait operation
        last_block_t1 = core.seq_sort(op_seq, key_fn=lambda b: b.t1)[-1].t1
        op_seq = core.seq_map_when(
            lambda b: isinstance(b, core.NamedBlock) and b.name == '_gap',
            lambda b: cmd.OperationBlock(  # make it a wait operation, as realistic as possible
                name='wait-until',
                t0=b.t0, t1=b.t1,
                subtype='gap',
                commands=[f"run.wait_until('{b.t1.isoformat()}')"],
                parameters={'t1': b.t1}
            ),
            core.seq_merge(core.NamedBlock(name='_gap', t0=t0, t1=last_block_t1), op_seq)
        )

        logger.info("simplifying hwp operations...")
        op_seq = simplify_hwp(op_seq)

        logger.info("------------- done! -------------")

        return op_seq

    def _apply_ops(self, state, op_cfgs, block=None):
        """
        Apply a series of operations to the current planning state, computing
        the updated state, the total duration, and resulting commands of the
        operations.

        Parameters
        ----------
        state : State
            The current planning state. It must be an object capable of tracking
            the mission's state and supporting time increment operations.
        op_cfgs : list of operation configs (dict)
            A list of operation configurations, where each configuration is a
            dictionary specifying the operation's parameters. Each operation
            configuration dict must have a 'sched_mode' key
        block : Optional[core.Block], optional
            per-block operations such as PreCal, PreObs, etc. require a block
            as part of the operation configuration.

        Returns
        -------
        new_state : State
            The new state after all operations have been applied.
        total_duration : int
            The total duration of all operations in seconds.
        commands : list of str
            A list of strings representing the commands generated by each
            operation. Commands are preconditioned by operation-specific
            indentation.

        """
        op_blocks = []
        duration = 0

        for op_cfg in op_cfgs:
            op_cfg = op_cfg.copy()

            # sanity check
            for k in ['name', 'sched_mode']:
                assert k in op_cfg, f"operation config must have a '{k}' key"

            # pop some non-operation kwargs
            op_name = op_cfg.pop('name')
            sched_mode = op_cfg.pop('sched_mode')
            indent = op_cfg.pop('indent', 0)        # n spaces for indentation
            divider = op_cfg.pop('divider', False)  # whether to add a divider
            # want_block = op_cfg.pop('want_block', True)  # whether to pass block in kwargs (if provided)

            # add block to the operation config if provided
            block_cfg = {'block': block} if block is not None else {}

            # one way to solve the problem of block parameter is to have an explicit option,
            # I commented it off to use a more lazy-man approach to get rid of it in make_op
            # block_cfg = {'block': block} if ((block is not None) and want_block) else {}

            op_cfg = {**op_cfg, **block_cfg}  # make copy

            # apply operation
            t_start = state.curr_time
            op = cmd.make_op(op_name, **op_cfg)
            state, dur, com = op(state)

            # add indentation if needed
            com = [f"{' '*indent}{c}" for c in com]

            # surround with empty lines
            # com = ['', *com, '']

            # add divider if needed
            if divider:
                com = ['# '+'-'*68, '#', f'# op mode: {sched_mode} op name: {op_name}', ''] + \
                       com + \
                      ['', '# '+'-'*68]

            # commands += com
            duration += dur
            state = state.increment_time(dt.timedelta(seconds=dur))
            op_blocks.append(cmd.OperationBlock(
                name=op_name,
                subtype=sched_mode,
                t0=t_start,
                t1=state.curr_time,
                commands=com,
                parameters=op_cfg
            ))

        return state, duration, op_blocks

    def _plan_block_operations(self, state, block, constraint, pre_ops, in_ops, post_ops):
        """
        Plan block operations based on the current state, block information, constraint, and operational sequences.

        Parameters
        ----------
        state : State
            Current state of the system.
        block : Block
            Block information containing start and end times.
        constraint : Constraint
            Constraint information containing start and end times.
        pre_ops : list
            List of pre-block operations.
        in_ops : list
            List of in-block operations.
        post_ops : list
            List of post-block operations.

        Returns
        -------
        State
            Updated state after planning the block operations.
        List[OperationBlock]
            Sequence of operations planned for the block.

        """
        # if we already pass the block or our constraint, nothing to do
        if state.curr_time >= block.t1 or state.curr_time >= constraint.t1:
            logger.debug(f"--> skipping block {block.name} because it's already past")
            return state, []

        # fast forward to within the constraint time block
        state = state.replace(curr_time=max(constraint.t0, state.curr_time))
        initial_state = state
        logger.debug(f"--> with constraint: planning {block.name} from {state.curr_time} to {block.t1}")

        op_seq = []

        # +++++++++++++++++++++
        # pre-block operations
        # +++++++++++++++++++++

        # need a multi-pass approach because we don't know a priori how long
        # the pre-block operations will take, so we will need a first pass to
        # get the right duration, and then a second pass to properly assign
        # start time for the operations
        i_pass = 0
        need_retry = True
        while need_retry:
            logger.debug(f"--> planning pre-block operations, pass {i_pass+1}")
            assert i_pass < 2, "needing more than two passes, unexpected, check code!"

            # did we already pass the block?
            t_start = state.curr_time
            if t_start >= block.t1:
                logger.debug(f"---> skipping block {block.name} because it's already past")
                return initial_state, []

            # opbs -> operation blocks
            state, duration, block_ops = self._apply_ops(state, pre_ops, block=block)

            logger.debug(f"---> pre-block ops duration: {duration} seconds")
            logger.debug(f"---> pre-block ops: {u.pformat(block_ops)}")
            logger.debug(f"---> pre-block curr state: {u.pformat(state)}")

            # what time are we starting?
            # -> start from t_start or block.t0-duration, whichever is later
            # -> overwrite block if we extended into the block
            # -> if we extended past the block, skip operation

            # did we extend into the block?
            if state.curr_time >= block.t0:
                logger.debug(f"---> curr_time extended into block {block.name}")
                # did we extend past entire block?
                if state.curr_time < block.t1:
                    logger.debug(f"---> curr_time did not extend past block {block.name}")
                    op_seq += block_ops
                    block = block.trim_left_to(state.curr_time)
                    logger.debug(f"---> trimmed block: {block}")
                    need_retry = False
            else:
                logger.debug(f"---> gap is large enough for pre-block operations")
                # if there is more time than we need to prepare for calibration,
                # let's wait to start later: need to regenerate commands with
                # a new t_start (time may have been encoded in the commands)
                state = initial_state.replace(curr_time=block.t0-dt.timedelta(seconds=duration))
                logger.debug(f"---> replanning with a later start time: {state.curr_time}")
                need_retry = True  # for readability

            logger.debug("---> need second pass? " + ("yes" if need_retry else "no"))
            i_pass += 1

        logger.debug(f"--> post pre-block state: {u.pformat(state)}")
        logger.debug(f"--> post pre-block op_seq: {u.pformat(op_seq)}")

        # +++++++++++++++++++
        # in-block operations
        # +++++++++++++++++++

        logger.debug(f"--> planning in-block operations from {state.curr_time} to {block.t1}")
        logger.debug(f"--> pre-planning state: {u.pformat(state)}")

        t_start = state.curr_time

        # skip if we are already past the block:
        # -> discard pre-block operations
        # -> revert to initial_state
        if t_start > block.t1:
            logger.debug(f"--> skipping in-block operations because we are already past the block")
            return initial_state, []

        # need a multi-pass approach because we don't know a priori how long
        # the post-block operations will take. If post-block operations run into
        # the boundary of the constraint, we will need to shrink the in-block
        # operations to fit within the given constraint
        i_pass = 0
        need_retry = True
        state_before = state
        while need_retry:
            logger.debug(f"--> planning in-block operations, pass {i_pass+1}")

            state, duration, op_in_block = self._apply_ops(state, in_ops, block=block)

            logger.debug(f"---> in-block ops duration: {duration} seconds")
            logger.debug(f"---> in-block ops: {u.pformat(op_in_block)}")
            logger.debug(f"---> in-block curr state: {u.pformat(state)}")

            # sanity check: if fail, it means post-cal operations are mixed into in-cal
            # operations
            assert state.curr_time <= block.t1, \
                "in-block operations are probably mixed with post-cal operations"

            # advance to the end of the block
            state = state.replace(curr_time=block.t1)

            logger.debug(f"---> post in-block state: {u.pformat(state)}")

            # +++++++++++++++++++++
            # post-block operations
            # +++++++++++++++++++++
            logger.debug(f"--> planning post-block operations, pass {i_pass+1}")
            t_start = state.curr_time

            state, duration, op_post_block = self._apply_ops(state, post_ops, block=block)

            logger.debug(f"---> post-block ops duration: {duration} seconds")
            logger.debug(f"---> post-block ops: {u.pformat(op_post_block)}")
            logger.debug(f"---> post-block curr state: {u.pformat(state)}")

            # have we extended past our constraint?
            if state.curr_time > constraint.t1:
                logger.debug(f"---> post-block ops extended past constraint")
                # shrink our block to make space for post-block operation and
                # revert to an old state before retrying
                block = block.shrink_right(state.curr_time - constraint.t1)
                state = state_before
                logger.debug(f"---> need to shrink block to: {block}")
                logger.debug(f"---> replanning from state: {state}")
                need_retry = True  # for readability
            else:
                op_seq += op_in_block
                op_seq += op_post_block
                need_retry = False

            logger.debug("---> need second pass? " + ("yes" if need_retry else "no"))

        return state, op_seq

    def _get_sun_safe_interval_for_block(self, block, state, t_start=None):
        """
        Find a sun-safe interval covering the block, starting from the current time
        provided by the state unless `t_start` is provided.

        Note that there is a complication with the ending time, because the scan can
        stop at any phase of the swipe, so it's hard to guarentee sun safety when we
        don't know its exact pointing. For now a suboptimal solution is to always
        restrict post-block operations to complete before the end of the block,
        because if the observation rules have been applied, we can be sure the block
        is sun safe in its duration.
        """
        if t_start is None: t_start = state.curr_time
        sun_rule = SunAvoidance(**self.rules['sun-avoidance'])
        sun_safe_covers = sun_rule(inst.StareBlock(name='_cover', t0=min(block.t0, t_start), t1=block.t1, az=block.az, alt=block.alt))
        safe_cover = [cover for cover in core.seq_flatten(sun_safe_covers) if cover.t0 <= block.t0 <= cover.t1]
        if len(safe_cover) == 0:
            logger.info(f"--> no sun-safe cover found for pre-block operations {block.name}")
            logger.info(f"--> constraining pre-block operations to start within block")
            constraint = core.Block(t0=block.t0, t1=block.t1)
        else:
            assert len(safe_cover) == 1, "unexpected number of sun safe covers"
            constraint = core.Block(t0=safe_cover[0].t0, t1=block.t1)
            logger.info(f"--> sun safe cover found: {constraint.t0.isoformat()} to {constraint.t1.isoformat()}")
        return constraint

    def _find_sun_safe_az_parking(self, t0, t1, az, el, time_step=5, angle_buffer=15):
        """find a good parking spot for our pointing when we want to do detector
        setup but our default az,el is sun unsafe. This function looks for a safe
        parking spot nearby."""
        sun_block = src.SourceBlock(name='sun', t0=t0, t1=t1, mode='both')
        t, az_sun, alt_sun = sun_block.get_az_alt(time_step=time_step)
        # make a fake scan over entire ranges and look for the time
        # when it is sufficiently away from the sun. Here we assume
        # the sun is moving very slowly, so min distance calculated
        # this way is reasonably close to the actual minimum, so
        # we can be quite safe after a buffer. 10x here is just to give
        # a reasonable speed that's faster than the sun: for 1h oper
        # this is 1 deg/s
        az_scan = np.mod(np.linspace(0, 10*360, len(t)), 360)
        alt_scan = az_scan*0 + el
        r = src.radial_distance(t, az_sun, alt_sun, az_scan, alt_scan)
        m = r >= (self.rules['sun-avoidance']['min_angle'] + angle_buffer)
        az_scan = az_scan[m]
        # calculate distance to original az: don't move too far away
        az_diff = (np.mod(az_scan-az, 360) + 180) % 360 - 180  # -180 -> 180
        az_choice = az_scan[np.argmin(np.abs(az_diff))]
        return az_choice

    def _revise_ops_constraint_with_maneuver(self, ops, constraint, block, state):
        """update (out-of-place) our operations and constraint if we intend to do az maneuver
        as we will need to automatically add additional `move_to` operations to avoid the sun.
        It will calculate new constraint and operations if a maneuver is possible, otherwise the
        old ops and constraint will passthrough. Additional information needs to be provided including
        the target scan block and the starting state.

        """
        # how much time do we have for pre-cal operation
        budget = (block.t0 - constraint.t0).total_seconds()

        # how much time do we need? run a first pass planning once to find out time pre-cal duration
        _, duration, _ = self._apply_ops(state, ops['pre_ops'], block)

        # are we limited causally? if so there is nothing we can do
        causal_limited = constraint.t0 == state.curr_time

        if duration > budget and (not causal_limited):
            logger.info("--> not enough time for pre-cal ops because of sun safety")

            # open up our constraint: do not break causal limit
            _t0 = state.curr_time
            _t1 = block.t0

            logger.info("--> need az maneuver: finding sun-safe az parking...")
            az_park = self._find_sun_safe_az_parking(_t0, _t1, block.az, block.alt)
            logger.info(f"--> found a sun-safe az parking: az={az_park:.2f}, el={block.alt}")

            # add moving ops for this block
            _pre_ops = [
                {'name': 'move_to', 'sched_mode': SchedMode.PreCal, 'az': az_park, 'el': block.alt},
                *ops["pre_ops"],
                {'name': 'move_to', 'sched_mode': SchedMode.PreCal, 'az': block.az, 'el': block.alt},
            ]
            # new constraint and ops for this block
            constraint = core.Block(t0=_t0, t1=block.t1)
            ops = ops | {"pre_ops": _pre_ops}
        else:
            logger.info("--> az maneuver not necessary")
        return ops, constraint

    def cmd2txt(self, op_seq):
        """
        Convert a sequence of operation blocks into a text representation.

        Parameters
        ----------
        op_seq : list of OperationBlock
            A sequence of operation blocks.

        Returns
        -------
        str
            A text representation of the sequence of operation blocks.

        """
        return '\n'.join(reduce(lambda x, y: x + y, [op.commands for op in op_seq], []))

    def build_schedule(self, t0: dt.datetime, t1: dt.datetime, state: State = None):
        """
        Run entire scheduling process to build a schedule for a given time range.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the schedule.
        t1 : datetime.datetime
            The end time of the schedule.
        state : Optional[State]
            The initial state of the observatory. If not provided, a default
            state will be initialized.

        Returns
        -------
        schedule as a text

        """
        # initialize sequences
        seqs = self.init_seqs(t0, t1)

        # apply observing rules
        seqs = self.apply(seqs)

        # initialize state
        state = state or self.init_state(t0)

        # plan operation seq
        op_seq = self.seq2cmd(seqs, t0, t1, state)

        # construct schedule str
        schedule = self.cmd2txt(op_seq)

        return schedule


# ------------------------
# utilities
# ------------------------

def round_robin(seqs_q, seqs_v=None, sun_avoidance=None, overlap_allowance=60*u.second):
    """
    Perform a round robin scheduling over sequences of time blocks, yielding non-overlapping blocks.

    This function goes through sequences of "query" time blocks (`seqs_q`) in a round robin fashion, checking for overlap
    between the blocks. An optional sequence of "value" time blocks (`seqs_v`) can be provided, which will be returned
    instead of the query blocks. The use case for having `seqs_v` different from `seqs_q` is that `seqs_q` can represent
    buffered time blocks used for determining overlap conditions, while `seqs_v`, representing the actual unbuffered time
    blocks, gets returned.

    Parameters
    ----------
    seqs_q : list of lists
        The query sequences. Each sub-list contains time blocks that are checked for overlap.
    seqs_v : list of lists, optional
        The value sequences. Each sub-list contains time blocks that are returned when their corresponding `seqs_q` block
        doesn't overlap with existing blocks.
    sun_avoidance : function / rule, optional
        If provided, a block is scheduled only if it satisfies this condition, this means the block is unchanged after
        the rule is applied.
    overlap_allowance: int
        minimum overlap to be considered in seconds, larger overlap will be rejected.

    Yields
    ------
    block
        Blocks from `seqs_v` that don't overlap with previously yielded blocks, as per the conditions defined.

    Notes
    -----
    This generator function exhaustively attempts to yield all non-overlapping time blocks from the provided sequences
    in a round robin order. The scheduling respects the order of sequences and the order of blocks within each sequence.
    It supports an optional sun avoidance condition to filter out undesirable time blocks based on external criteria
    (for example, blocks that are in direct sunlight).

    Examples
    --------
    >>> seqs_q = [[block1, block2], [block3]]
    >>> list(round_robin(seqs_q))
    [block1, block3, block2]

    """
    if seqs_v is None:
        seqs_v = seqs_q
    assert len(seqs_q) == len(seqs_v)

    n_seq = len(seqs_q)
    seq_i = 0
    block_i = [0] * n_seq

    merged = []
    while True:
        # return if we have exhausted all scans in all seqs
        if all([block_i[i] >= len(seqs_q[i]) for i in range(n_seq)]):
            return

        # cycle through seq -> add the latest non-overlaping block -> continue to next seq
        # skip if we have exhaused all scans in a sequence
        if block_i[seq_i] >= len(seqs_q[seq_i]):
            seq_i = (seq_i + 1) % n_seq
            continue

        seq_q = seqs_q[seq_i]
        seq_v = seqs_v[seq_i]
        block_q = seq_q[block_i[seq_i]]
        block_v = seq_v[block_i[seq_i]]

        # can we schedule this block?
        #  yes if:
        #  - it doesn't overlap with existing blocks
        #  - it satisfies sun avoidance condition if specified
        overlap_ok = not core.seq_has_overlap_with_block(merged, block_q, allowance=overlap_allowance)
        if not overlap_ok:
            logger.info(f"-> Block {block_v} overlaps with existing block, skipping")

        if sun_avoidance is not None:
            sun_ok = block_q == sun_avoidance(block_q)
            if not sun_ok:
                logger.info(f"-> Block {block_v} fails sun check, skipping")

        ok = overlap_ok * sun_ok
        if ok:
            # schedule and move on to next seq
            yield block_v
            merged += [block_q]
            seq_i = (seq_i + 1) % n_seq

        block_i[seq_i] += 1

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
                commands=[f"run.wait_until('{b_next.t1.isoformat()}')"]
            )]
        else:
            return seq_prev+[b_next]
    return reduce(rewriter, op_seq, [])
