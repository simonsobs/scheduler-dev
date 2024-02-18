"""A production-level implementation of the SAT policy

"""
import yaml
import os.path as op
from dataclasses import dataclass, field, replace as dc_replace
import datetime as dt
from typing import List, Union, Optional, Dict
import numpy as np
from collections import Counter
from enum import Enum
import jax.tree_util as tu

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u

logger = u.init_logger("sat-policy")

class SchedMode(Enum):
    """
    Enumerates the scheduling modes for satellite operations.

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
class State:
    """Observatory state relevant for operation planning.
    Made immutable for the piece of mind.

    """
    curr_time: dt.datetime
    az_now: float
    el_now: float
    hwp_spinning: bool
    boresight_rot_now: Optional[float] = None
    # prev_state: Optional["State"] = None

    def replace(self, **kwargs):
        # enble entire state history for debugging
        # kwargs = {**kwargs, "prev_state": self}
        return dc_replace(self, **kwargs)

# ====================
# register operations
# ====================

@cmd.operation(name="preamble", duration=0)
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
    "    time.sleep(5)",
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
    "    time.sleep(5)",
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
    "        pid.acq.start()",
    "",
    "        time.sleep(5)",
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
    "def HWPFastStop():",
    f"    iboot2 = OCSClient('{hwp_cfg['iboot2']}')",
    f"    pid = OCSClient('{hwp_cfg['pid']}')",
    f"    pmx = OCSClient('{hwp_cfg['pmx']}')",
    "    pid.acq.stop()",
    "    time.sleep(5)",
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
    "        pid.acq.start()",
    "",
    "        time.sleep(5)",
    "        start_freq = float(pid.acq.status().session['data']['current_freq'])",
    "        time.sleep(15)",
    "        cur_freq = float(pid.acq.status().session['data']['current_freq'])",
    "        if cur_freq > start_freq:",
    "            if forward:",
    "                pid.set_direction(direction = '0')",
    "            else:",
    "                pid.set_direction(direction = '1')",
    "",
    "            start_freq = cur_freq",
    "            time.sleep(15)",
    "            cur_freq = float(pid.acq.status().session['data']['current_freq'])",
    "            if cur_freq > start_freq:",
    "                pmx.set_off()",
    "                iboot2.set_outlet(outlet = 1, state = 'off')",
    "                iboot2.set_outlet(outlet = 2, state = 'off')",
    "                time.sleep(60*30)",
    "",
    "        while cur_freq > 0.2:",
    "            cur_freq = float(pid.acq.status().session['data']['current_freq'])",
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

@cmd.operation(name='wrap-up')
def wrap_up(state):
    state = state.update(az_now=180, el_now=48)
    return state, [
        "# go home",
        "run.acu.move_to(az=180, el=48)",
        "",
        "time.sleep(1)"
    ]

@cmd.operation(name='ufm-relock', return_duration=True)
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
            "",
            "run.smurf.uxm_relock(concurrent=True)",
        ]
    else:
        return state, 0, ["# no ufm relock needed at this time"]

@cmd.operation(name='hwp-spin-up', return_duration=True)
def hwp_spin_up(state, disable_hwp):
    if not disable_hwp and not state.hwp_spinning:
        state = state.replace(hwp_spinning=True)
        return state, 20*u.minute, [
            "HWPPrep()",
            "forward = True",
            "hwp_freq = 2.0",
            "HWPSpinUp()",
        ]
    return state, 0, ["# hwp disabled or already spinning"]

@cmd.operation(name='hwp-spin-down', return_duration=True)
def hwp_spin_down(state, disable_hwp):
    if not disable_hwp and state.hwp_spinning:
        state = state.replace(hwp_spinning=False)
        return state, 10*u.minute, [
            "HWPFastStop()",
            "HWPPost()",
            "hwp_freq = 0.0",
        ]
    return state, 0, ["# hwp disabled or not spinning"]

@cmd.operation(name='set-scan-params', duration=0)
def set_scan_params(state, az_speed, az_accel):
    if az_speed != state.az_speed_now or az_accel != state.az_accel_now:
        state = state.replace(az_speed_now=az_speed, az_accel_now=az_accel)
        return state, [
            f"run.acu.set_scan_params({az_speed}, {az_accel})",
        ]
    return state, []

# per block operation: block will be passed in as parameter
@cmd.operation(name='det-setup', return_duration=True)
def det_setup(state, block, disable_hwp=False):
    # only do it if boresight has changed
    duration = 0
    commands = []
    if block.alt != state.el_now:
        if not disable_hwp and state.hwp_spinning:
            state, d, c = hwp_spin_down(state)
            commands += c
            duration += d
        commands += [
            "",
            f"run.wait_until('{block.t0.isoformat()}')"
            "################### Detector Setup######################",
            f"run.acu.move_to(az={round(block.az, 3)}, el={round(block.alt,3)})",
            "run.smurf.take_bgmap(concurrent=True)",
            "run.smurf.iv_curve(concurrent=False, settling_time=0.1)",
            "run.smurf.bias_dets(concurrent=True)",
            "time.sleep(180)",
            "run.smurf.bias_step(concurrent=True)",
            "#################### Detector Setup Over ####################",
            "",
        ]
        duration += 60
        if not disable_hwp and not state.hwp_spinning:
            state, d, c = hwp_spin_up(state)
            commands += c
            duration += d

    return state, duration, commands

@cmd.operation(name='setup-boresight', duration=0)  # TODO check duration
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    if apply_boresight_rot and state.boresight_rot_now != block.boresight_rot:
        commands += [f"run.acu.set_boresight({block.boresight_angle}"]
        state = state.replace(boresight_rot_now=block.boresight_rot)

    if block.az != state.az_now or block.alt != state.el_now:
        commands += [ f"run.acu.move_to(az={round(block.az,3)}, el={round(block.alt,3)})" ]
        state = state.replace(az_now=block.az, el_now=block.alt)
    return state, commands

@cmd.operation(name='cmb-scan', return_duration=True)
def cmb_scan(block):
    commands = [
        "run.seq.scan(",
        f"    description='{block.name}',",
        f"    stop_time='{block.t1.isoformat()}',",
        f"    width={round(block.throw,3)}, az_drift=0,",
        f"    subtype='cmb', tag='{block.tag}',",
        ")",
    ]
    return block.duration, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='bias-det', duration=60)
def bias_det(*args, **kwargs):
    return [
        "run.smurf.bias_dets(concurrent=True)"
    ]

@cmd.operation(name='source-scan', return_duration=True)
def source_scan(state, block):
    block = block.trim_left_to(state.curr_time)
    if block is None:
        return 0, ["# too late, don't scan"]
    state = state.replace(az_now=block.az, el_now=block.alt)
    return state, block.duration.total_seconds(), [
        "################# Scan #################################",
        "",
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
        "",
        "################# Scan Over #############################",
    ]

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
    cal_targets : list
        a list of tuples, each tuple specifies a calibration target, with the format
        (source, array_query, el_bore, boresight_rot, tagname)
    cal_policy : str
    scan_tag : str
        a tag to be added to all scans
    az_speed : float
        the az speed in deg / s
    az_accel : float
        the az acceleration in deg / s^2
    apply_boresight_rot : bool
        whether to apply boresight rotation
    allow_partial : bool
        whether to allow partial source scans
    wafer_sets : dict[str, str]
        a dict of wafer sets definitions
    preamble_file : str
        a file containing preamble commands to be executed before the start of the sequence
    checkpoints : dict
        a dict of checkpoints, with keys being checkpoint names and values being blocks
        for internal bookkeeping
    """
    blocks: dict
    rules: Dict[str, core.Rule]
    geometries: List[dict]
    cal_targets: List[tuple]
    cal_policy: str = 'round-robin'
    scan_tag: Optional[str] = None
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s^2
    apply_boresight_rot: bool = False
    allow_partial: bool = False
    wafer_sets: dict[str, str] = field(default_factory=dict)
    preamble_file: Optional[str] = None
    operations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Union[dict, str]):
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
            source = cal_target[0]
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

        # give some feedbacks to the user
        c = Counter(core.seq_map(lambda x: type(x), core.seq_flatten(blocks)))
        logger.info(f"Number of blocks initialized: {dict(c)}")

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
        assert 'sun-avoidance' in self.rules
        sun_rule = ru.make_rule('sun-avoidance', **self.rules['sun-avoidance'])
        blocks['calibration'] = sun_rule(blocks['calibration'])
        
        # -----------------------------------------------------------------
        # step 2: plan calibration scans
        #   - refer to each target specified in cal_targets
        #   - same source can be observed multiple times with different
        #     array configurations (i.e. using array_query)
        # -----------------------------------------------------------------
        cal_blocks = []

        for cal_target in self.cal_targets:
            source, array_query, el_bore, boresight_rot, tagname = cal_target
            assert source in blocks['calibration'], f"source {source} not found in sequence"

            # digest array_query: it could be a fnmatch pattern matching the path
            # in the geometry dict, or it could be looked up from a predefined
            # wafer_set dict. Here we account for the latter case:
            # look up predefined query in wafer_set
            if array_query in self.wafer_sets:
                array_query = self.wafer_sets[array_query]

            # build array geometry information based on the query
            array_info = inst.array_info_from_query(self.geometries, array_query)

            # apply MakeCESourceScan rule to transform known observing windows into
            # actual scans
            rule = ru.MakeCESourceScan(
                array_info=array_info, 
                el_bore=el_bore, 
                drift=True,
                boresight_rot=boresight_rot,
                allow_partial=self.allow_partial,
            )
            source_scans = rule(blocks['calibration'][source])

            # add tags to the scans
            cal_blocks.append(
                core.seq_map(
                    lambda block: block.replace(tag=f"{block.tag},{tagname}"), 
                    source_scans
                )
            )

        # -----------------------------------------------------------------
        # step 3: resolve calibration target conflicts 
        #   currently we adopt a simple round-robin strategy to resolve
        #   conflicts between multiple calibration targets. This is done
        #   by cycling through the calibration targets and add scan blocks
        #   successively in the order given in the cal_targets config.
        # -----------------------------------------------------------------

        try:
            # currently only implemented round-robin approach, but can be extended
            # to other strategies
            cal_policy = {
                'round-robin': round_robin
            }[self.cal_policy]
        except KeyError:
            raise ValueError(f"unsupported calibration policy: {self.cal_policy}")

        # done with the calibration blocks
        blocks['calibration'] = list(cal_policy(
            cal_blocks, 
            sun_avoidance=sun_rule
        ))

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

    def init_state(self, t0):
        """This function provides some reasonable guess for the initial state, but in practice, 
        it should be supplied by the observatory controller.

        """
        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            last_ufm_relock=None,
            hwp_spinning=False,
            az_speed_now=None,
            az_accel_now=None
        )

    def seq2cmd(self, seq: core.Blocks, t0: dt.datetime):
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

        Returns
        -------
        list of str
            A list of command strings that will be executed by the telescope

        """
        # -----------------------------------------------------------------
        # 1. pre-session operations
        # -----------------------------------------------------------------

        op_blocks = []
        commands = []
        state = self.init_state(t0)

        ops = [op for op in self.operations if op.pop('sched_mode') == SchedMode.PreSession]
        for op_cfg in ops:
            sched_mode = op_cfg.pop('sched_mode')
            op = cmd.make_op(**op_cfg)
            dur, com = op(state)
            logger.info(f"planning operation: mode: {sched_mode} name: {op_cfg['name']} duration: {dur} seconds")

            # add comment and some empty lines for readability
            commands += ['', f'# op mode: {sched_mode} op name: {op_cfg["name"]}'] + com + ['']
            state['curr_time'] += dt.timedelta(seconds=dur)

        op_blocks += [core.NamedBlock(name='init', t0=t0, t1=state['curr_time'])]
        post_init_state = state

                        )


        return cmd.CompositeCommand(commands)


# ------------------------
# utilities
# ------------------------
def round_robin(seqs_q, seqs_v=None, sun_avoidance=None):
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
    >>> seqs_q = [[[1, 2], [3, 4]], [[5, 6]]]
    >>> list(round_robin(seqs_q))
    [[1, 2], [5, 6], [3, 4]]

    >>> def avoid_sun(block):
    ...     return block if block[0] % 2 == 0 else block
    >>> seqs_q = [[[1,3], [2, 4]], [[6, 7]]]
    >>> seqs_v = [[[10, 15], [20, 25]], [[30, 35]]]
    >>> list(round_robin(seqs_q, seqs_v=seqs_v, sun_avoidance=avoid_sun))
    [[20, 25], [30, 35]]

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
        ok = not core.seq_has_overlap_with_block(merged, block_q)
        if sun_avoidance is not None:
            ok *= block_q == sun_avoidance(block_q)
        
        if ok:
            # schedule and move on to next seq
            yield block_v
            merged += [block_q]
            seq_i = (seq_i + 1) % n_seq
        else:
            # unsuccess, retry with next block
            logger.info(f"Calibration block {block_v} overlaps with existing blocks, skipping...")

        block_i[seq_i] += 1
