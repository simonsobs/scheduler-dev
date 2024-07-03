"""
Base telescope policy for SAT and LAT.

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

logger = u.init_logger(__name__)

@dataclass(frozen=True)
class State(cmd.State):
    """
    State relevant to telescope operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.

    Parameters
    ----------
    boresight_rot_now : Optional[float]
        The current boresight rotation state.
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    """
    boresight_rot_now: Optional[float] = 0
    last_ufm_relock: Optional[dt.datetime] = None
    last_bias_step: Optional[dt.datetime] = None
    last_iv: Optional[dt.datetime] = None
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
#                Operation Helpers
# ----------------------------------------------------
# Note that we are not registering here
# These are helper functions for the LAT and SAT to use in there operations

def preamble():
    return [
    "from nextline import disable_trace",
    "import time",
    "import datetime",
    "",
    "with disable_trace():",
    "    import numpy as np",
    "    import sorunlib as run",
    "    from ocs.ocs_client import OCSClient",
    "    run.initialize()",
    "",
    "UTC = datetime.timezone.utc",
    "acu = run.CLIENTS['acu']",
    "pysmurfs = run.CLIENTS['smurf']",
    "",
    ]

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
            "############# Daily Relock",
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

# per block operation: block will be passed in as parameter
def det_setup(state, block, apply_boresight_rot=True, iv_cadence=None):
    # when should det setup be done?
    # -> should always be done if the block is a cal block
    # -> should always be done if elevation has changed
    # -> should always be done if det setup has not been done yet
    # -> should be done at a regular interval if iv_cadence is not None
    # -> should always be done if boresight rotation has changed
    doit = (block.subtype == 'cal') or (block.alt != state.el_now) 
    doit = doit or (not state.is_det_setup) 
    doit = doit or (iv_cadence is not None and ((state.last_iv is None) or ((state.curr_time - state.last_iv).total_seconds() > iv_cadence)))  
    if apply_boresight_rot and (block.boresight_angle != state.boresight_rot_now):
        doit = True

    if doit:
        commands = [
            "",
            "################### Detector Setup######################",
            "run.smurf.take_bgmap(concurrent=True)",
            "run.smurf.iv_curve(concurrent=True, ",
            "    iv_kwargs={'run_serially': False, 'cool_wait': 60*5})",
            "run.smurf.bias_dets(concurrent=True)",
            "time.sleep(180)",
            "run.smurf.bias_step(concurrent=True)",
            "#################### Detector Setup Over ####################",
            "",
        ]
        state = state.replace(
            last_bias_step=state.curr_time,
            is_det_setup=True,
            last_iv = state.curr_time,
        )
        return state, 12*u.minute, commands
    else:
        return state, 0, []

def cmb_scan(state, block):
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

    commands.extend([
        f"scan_stop = {repr(block.t1)}",
        f"if datetime.datetime.now(tz=UTC) < scan_stop - datetime.timedelta(minutes=10):",
        "    run.seq.scan(",
        f"        description='{block.name}',",
        f"        stop_time='{block.t1.isoformat()}',",
        f"        width={round(block.throw,3)}, az_drift=0,",
        f"        subtype='cmb', tag='{block.tag}',",
        "    )",
    ])
    return state, (block.t1 - state.curr_time).total_seconds(), commands

def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0

    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += 1*u.minute

    return state, duration, commands

# passthrough any arguments, to be used in any sched-mode
def bias_step(state, min_interval=10*u.minute):
    if state.last_bias_step is None or (state.curr_time - state.last_bias_step).total_seconds() > min_interval:
        state = state.replace(last_bias_step=state.curr_time)
        return state, 60, [ "run.smurf.bias_step(concurrent=True)", ]
    else:
        return state, 0, []

def wrap_up(state, az_stow, el_stow):
    state = state.replace(az_now=az_stow, el_now=el_stow)
    return state, [
        "# go home",
        f"run.acu.move_to(az={az_stow}, el={el_stow})",
        "time.sleep(1)"
    ]

@dataclass
class TelPolicy:
    """Base telescope policy for LAT and SAT.

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
    iso_scan_speeds: Optional =None
    boresight_override: Optional[float] = None
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s^2
    allow_az_maneuver: bool = True
    wafer_sets: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    stages: Dict[str, Any] = field(default_factory=dict)

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
                blocks = inst.parse_sequence_from_toast(loader_cfg['file'])
                if self.boresight_override is not None:
                    blocks = core.seq_map(
                        lambda b: b.replace(
                            boresight_angle=self.boresight_override
                        ), blocks
                    )
                return blocks
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
                lambda block: block.replace(
                                az_speed = self.az_speed,
                                az_accel = self.az_accel,
                                tag=f"{block.tag},{target.tag}"
                            ),
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

        blocks = core.seq_sort(blocks['baseline']['cmb'] + blocks['calibration'], flatten=True)

        # alternate scan speeds for ISO testing
        # run after flattening to presume order is set
        if self.iso_scan_speeds is not None:
            self.c = 0
            def iso_set(block):
                if block.subtype != 'cmb':
                    return block
                v,a = self.iso_scan_speeds[ self.c % len(self.iso_scan_speeds)]
                self.c += 1
                return block.replace(az_speed=v, az_accel=a)
            blocks = core.seq_map(iso_set, blocks)
            del self.c

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
        )

    def seq2cmd(
        self, 
        seq, 
        t0: dt.datetime, 
        t1: dt.datetime, 
        state: Optional[State] = None
    ) -> List[Any]:
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
        list of Operation

        """
        if state is None:
            state = self.init_state(t0)

        # load building stage
        build_op = get_build_stage('build_op', **self.stages.get('build_op', {}))
        ops = build_op.apply(seq, t0, t1, state, self.operations)
        return ops

    def cmd2txt(self, irs, t0, t1, state=None):
        """
        Convert a sequence of operation blocks into a text representation.

        Parameters
        ----------
        irs : list of IR
            A sequence of operation blocks.

        Returns
        -------
        str
            A text representation of the sequence of operation blocks.

        """
        if state is None:
            state = self.init_state(t0)
        build_sched = get_build_stage('build_sched', **self.stages.get('build_sched', {}))
        commands = build_sched.apply(irs, t0, t1, state)
        return '\n'.join(commands)

    def build_schedule(self, t0: dt.datetime, t1: dt.datetime, state: Optional[State] = None):
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
        ir = self.seq2cmd(seqs, t0, t1, state)

        # construct schedule str
        schedule = self.cmd2txt(ir, t0, t1, state)

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

        sun_ok = True
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
