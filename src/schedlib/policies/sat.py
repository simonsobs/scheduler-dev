"""A production-level implementation of the SAT policy

"""
import yaml
import os.path as op
from dataclasses import dataclass, field
import datetime as dt
from typing import List, Union, Optional, Dict, Any
import jax.tree_util as tu
from functools import reduce

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..commands import SchedMode, State

logger = u.init_logger(__name__)

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
    allow_partial : bool
        whether to allow partial source scans
    wafer_sets : dict[str, str]
        a dict of wafer sets definitions
    operations : List[Dict[str, Any]]
        an orderred list of operation configurations
    """
    blocks: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, core.Rule] = field(default_factory=dict)
    geometries: List[Dict[str, Any]] = field(default_factory=list)
    cal_targets: List[Any] = field(default_factory=list)
    cal_policy: str = 'round-robin'
    scan_tag: Optional[str] = None
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s^2
    allow_partial: bool = False
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
            sun_rule = ru.make_rule('sun-avoidance', **self.rules['sun-avoidance'])
            blocks = sun_rule(blocks)
        else:
            logger.warning("no sun avoidance rule specified!")
            sun_rule = None

        # -----------------------------------------------------------------
        # step 2: plan calibration scans
        #   - refer to each target specified in cal_targets
        #   - same source can be observed multiple times with different
        #     array configurations (i.e. using array_query)
        # -----------------------------------------------------------------
        logger.info("planning calibration scans...")
        cal_blocks = []

        for cal_target in self.cal_targets:
            logger.info(f"-> planning calibration scans for {cal_target}...")
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
            logger.debug(f"-> array_info: {array_info}")

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
            source_scans = core.seq_flatten(source_scans)

            # add tags to the scans
            cal_blocks.append(core.seq_map(
                lambda block: block.replace(tag=f"{block.tag},{tagname}"),
                source_scans
            ))

            logger.info(f"-> found {len(source_scans)} scan options for {source}: {u.pformat(source_scans)}")

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
        blocks['calibration'] = list(cal_policy(cal_blocks, sun_avoidance=sun_rule))

        # check sun avoidance again
        blocks['calibration'] = core.seq_flatten(sun_rule(blocks['calibration']))

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
        raise NotImplementedError("init_state should be implemented in each subclass")

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

        logger.debug(f"post-init op_seq: {u.pformat(op_seq)}")
        logger.info(f"post-init state: {u.pformat(state)}")

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

        cal_blocks = core.seq_sort(seq['calibration'], flatten=True)

        pre_ops  = [op for op in self.operations if op['sched_mode'] == SchedMode.PreCal]
        in_ops   = [op for op in self.operations if op['sched_mode'] == SchedMode.InCal]
        post_ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostCal]

        cal_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        logger.debug(f"cal_ops to plan: {cal_ops}")
        logger.debug(f"pre-cal state: {u.pformat(state)}")

        for block in cal_blocks:
            logger.info(f"-> planning cal block: {u.pformat(block)}")

            # skip if we are already past the block
            if state.curr_time >= block.t1:
                logger.info(f"--> skipping cal block {block.name} because it's already past")
                continue

            # constraint: calibration blocks take highest priority. So, for simplicity
            # we give no a priori constraint for the starting time by setting it at
            # the beginning of the time range. For the ending time, because the scan can
            # stop at any phase of the swipe, it's hard to guarentee sun safety when we 
            # don't know its exact pointing. There might be a good solution to this but
            # for now a simple fix is to constrain post-cal operations to complete 
            # before the end of the block, as we can be sure that the block is sun safe 
            # after observation rules are applied.

            # what's our constraint from sun safety?
            # -> we treat the time prior to the block as a stare scan (throw=0)
            sun_rule = ru.make_rule('sun-avoidance', **self.rules['sun-avoidance'])
            sun_safe_covers = sun_rule(inst.StareBlock(name='_cover', t0=min(block.t0, state.curr_time), t1=block.t1, az=block.az, alt=block.alt))
            logger.info(f"--> sun-safe covers: {u.pformat(sun_safe_covers)}")
            safe_cover = [cover for cover in core.seq_flatten(sun_safe_covers) if cover.t0 <= block.t0 <= cover.t1]
            if len(safe_cover) == 0:
                logger.info(f"--> no sun-safe cover found for block {block.name}")
                logger.info(f"--> constraining pre-cal operations to start within block")
                constraint = core.Block(t0=block.t0, t1=block.t1)
            else:
                assert len(safe_cover) == 1, "unexpected number of sun safe covers"
                constraint = core.Block(t0=safe_cover[0].t0, t1=block.t1)
                logger.info(f"--> sun safe cover found: {constraint.t0.isoformat()} to {constraint.t1.isoformat()}")

            # plan pre-, in-cal, post-cal operations for the block under constraint
            # -> returns the new state, and a list of OperationBlock
            state, block_ops = self._plan_block_operations(state, block, constraint, **cal_ops)

            logger.debug(f"--> post-block ops: {u.pformat(block_ops)}")
            logger.debug(f"--> post-block state: {u.pformat(state)}")

            op_seq += block_ops

        post_cal_state = state
        # logger.debug(f"post-cal op_seq: {u.pformat(op_seq)}")
        logger.debug(f"post-cal state: {u.pformat(state)}")

        # -----------------------------------------------------------------
        # 3. cmb scans
        #
        # Note: for cmb scans, we will avoid overlapping with calibrations;
        # this means we will tend to overwrite into cmb blocks more often
        # -----------------------------------------------------------------
        logger.info("---------- step 3: planning cmb ops ----------")

        # calibration always take precedence, so we remove the overlapping region
        # from cmb scans first: this is done by first merging cal ops into cmb seq
        # and then filtering out non-cmb blocks
        cmb_blocks = core.seq_flatten(core.seq_filter(
            lambda b: isinstance(b, inst.ScanBlock) and b.subtype == 'cmb',
            core.seq_merge(seq['baseline']['cmb'], op_seq, flatten=True)
        ))

        pre_ops  = [op for op in self.operations if op['sched_mode'] == SchedMode.PreObs]
        in_ops   = [op for op in self.operations if op['sched_mode'] == SchedMode.InObs]
        post_ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostObs]

        cmb_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        # assume we are starting from the end of initialization
        state = post_init_state

        logger.debug(f"cmb_ops to plan: {cmb_ops}")
        logger.debug(f"pre-planning state: {state}")

        # avoid pre-cmb operations from colliding with calibration, so we will
        # fast-forward our watch to the closest non-colliding time
        full_interval = core.NamedBlock(name='_dummy', t0=t0, t1=t1)
        non_overlapping = core.seq_flatten(core.seq_filter(
            lambda b: b.name == '_dummy',
            core.seq_merge([full_interval], op_seq, flatten=True)
        ))

        for block in cmb_blocks:
            logger.info(f"-> planning cmb block: {u.pformat(block)}")
            # skip if we are already past the block
            if state.curr_time >= block.t1:
                logger.debug(f"--> skipping cmb block {block.name} because it's already past")
                continue

            # look for covering non-overlapping interval
            constraint = [b for b in non_overlapping if b.t0 <= block.t0 and block.t1 <= b.t1][0]
            logger.info(f"--> operational constraint: {constraint.t0} to {constraint.t1}")


            # plan pre- / in- / post-cmb operations for the block under constraint
            state, block_ops = self._plan_block_operations(state, block, constraint, **cmb_ops)

            logger.debug(f"--> post-block ops: {u.pformat(block_ops)}")
            logger.debug(f"--> post-block state: {u.pformat(state)}")

            op_seq += block_ops

        post_cmb_state = state
        # logger.debug(f"post-cmb op_seq: {u.pformat(op_seq)}")
        logger.debug(f"post-cmb state: {state}")

        # -----------------------------------------------------------------
        # 4. post-session operations
        # -----------------------------------------------------------------
        logger.info("---------- step 4: planning post-session ops ----------")

        # decide whether last scan is cmb or calibration
        # note: this assumes cmb and cal operations are independent -> might be too optimistic
        # possible solution: go through op_seq to evolve the state fully -> task for another day
        state = post_cmb_state if post_cmb_state.curr_time > post_cal_state.curr_time else post_cal_state

        ops = [op for op in self.operations if op['sched_mode'] == SchedMode.PostSession]
        state, _, post_session_ops = self._apply_ops(state, ops)
        op_seq += post_session_ops

        logger.debug(f"post-session state: {state}")
        logger.debug(f"post-session ops: {u.pformat(post_session_ops)}")

        logger.debug("---------- finished planning ----------")

        # make sure operations are sorted by time, and we want to do that safely
        # without disturbing the original order of operations especially when
        # we have lots of no duration operations that look like they start at
        # the same time
        op_seq = core.seq_sort(op_seq)

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
                logger.debug(f"---> replanning from state: {u.pformat(state)}")
                need_retry = True  # for readability
            else:
                op_seq += op_in_block
                op_seq += op_post_block
                need_retry = False

            logger.debug("---> need second pass? " + ("yes" if need_retry else "no"))

        return state, op_seq

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
            logger.info(f"Calibration block {block_v} overlaps with existing blocks or fails sun check, skipping...")

        block_i[seq_i] += 1
