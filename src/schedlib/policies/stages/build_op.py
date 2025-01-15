"""Optimization pass that resolves overlapping of operations and
make prioritization between calibration sequence and baseline sequence.

It begins by converting a sequence of ScanBlock into an intermediate
representation with each block surrounded by operation blocks.
This representation will be subject to several optimization at this
level without actually being lowered into commands.

"""
import numpy as np
import datetime as dt
import copy
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field, replace as dc_replace
from schedlib import core, commands as cmd, utils as u, rules as ru
from schedlib.thirdparty.avoidance import get_sun_tracker
from schedlib.commands import SchedMode

logger = u.init_logger(__name__)

def get_traj_ok_time(az0, az1, alt0, alt1, t0, sun_policy):
    # Returns the timestamp until which the move from
    # (az0, alt0) to (az1, alt1) is sunsafe.
    sun_tracker = get_sun_tracker(u.dt2ct(t0), policy=sun_policy)
    az = np.linspace(az0, az1, 101)
    el = np.linspace(alt0, alt1, 101)
    if alt0 != alt1 or az0 != az1:
        az1 = (az[:,None] + el[None,:]*0).ravel()
        el1 = (az[:,None]*0 + el[None,:]).ravel()
        az, el = az1, el1
    sun_safety = sun_tracker.check_trajectory(t=u.dt2ct(t0), az=az, el=el)
    return u.ct2dt(u.dt2ct(t0) + sun_safety['sun_time'])

def get_parking(t0, t1, alt0, sun_policy, az_parking=180):
    # gets a safe parking location for the time range and
    # Do the move in two steps, parking at az_parking (180
    # is most likely to be sun-safe).  Identify a spot that
    # is safe for the duration of t0 to t1.
    alt_range = alt0, sun_policy['min_el']
    n_alts = max(2, int(round(abs(alt_range[1] - alt_range[0]) / 4. + 1)))
    for alt_parking in np.linspace(alt_range[0], alt_range[1], n_alts):
        safet = get_traj_ok_time(az_parking, az_parking, alt_parking, alt_parking,
                                    t0, sun_policy)
        if safet >= t1:
            break
    else:
        raise ValueError(f"Sun-safe parking spot not found. az {az_parking} "
                            f"el {alt_parking} is safe only until {safet}")

    # Now bracket the moves, hopefully with ~5 minutes on each end.
    buffer_t = min(300, int((t1 - t0).total_seconds() / 2))
    t0_parking = t0 + dt.timedelta(seconds=buffer_t)
    t1_parking = t1 - dt.timedelta(seconds=buffer_t)

    return az_parking, alt_parking, t0_parking, t1_parking

# some additional auxilary command classes that will be mixed 
# into the IR to represent some intermediate operations. They
# don't need to contain all the fields of a regular IR
@dataclass(frozen=True)
class Aux: pass

@dataclass(frozen=True)
class MoveTo(Aux):
    az: float
    alt: float
    subtype: str = "aux"
    def __repr__(self):
        return f"# move to az={self.az:.2f}"

@dataclass(frozen=True)
class WaitUntil(Aux):
    t1: dt.datetime
    az: float
    alt: float
    subtype: str = "aux"
    def __repr__(self):
        return f"# wait until {self.t1} at az = {self.az:.2f}"

# full intermediate representation of operation used in this
# build stage
@dataclass(frozen=True)
class IR(core.Block):
    name: str
    subtype: str
    t0: dt.datetime
    t1: dt.datetime
    az: float
    alt: float
    block: Optional[core.Block] = field(default=None, hash=False, repr=False)
    operations: List[Dict[str, Any]] = field(default_factory=list, hash=False, repr=False)

    def __repr__(self):
        az = f"{self.az:>7.2f}" if self.az is not None else f"{'None':>7}"
        return f"{self.name[:15]:<15} ({self.subtype[:8]:<8}) az = {az}: {self.t0.strftime('%y-%m-%d %H:%M:%S')} -> {self.t1.strftime('%y-%m-%d %H:%M:%S')}"

    def replace(self, **kwargs):
        """link `replace` in the wrapper block with the block it contains.
        Note that when IR is produced, we assume no trimming needs to happen,
        so we use `dc_replace` instead of `super().replace` which accounts for
        trimming effect on drift scans. It is not necessary here as we are
        merely solving for different unwraps for drift scan.

        We allow the option of changing the block or block.subtype here because 
        sometimes we need to run a master schedule but mark things as
        calibration. Most important example: drone calibration campaigns 
        """
        if self.block is not None and 'block' not in kwargs:
            block_kwargs = {k: v for k, v in kwargs.items() if k in ['t0', 't1', 'az', 'alt', 'subtype']}
            new_block = dc_replace(self.block, **block_kwargs)
            kwargs['block'] = new_block
        return dc_replace(self, **kwargs)

# some mode enums relevant for IR building in this stage
# didn't use built-in enum in python as they turn things into
# objects and not str.
class IRMode:
    PreSession = 'pre_session'
    PreBlock = 'pre_block'
    InBlock = 'in_block'
    PostBlock = 'post_block'
    PostSession = 'post_session'
    Gap = 'gap'
    Aux = 'aux'


@dataclass(frozen=True)
class BuildOp:
    """
    BuildOp represents the stage that converts ScanBlock into a schedule of
    Operations (called intermediate representation, or `IR`, in this script).

    Attributes
    ----------
    policy_config : Dict[str, Any]
        Full configuration of the SATPolicy.
    min_duration : float
        Parameter for min-duration rule: minimum duration of a block to schedule.
    min_cmb_duration: float
        Parameter for second pass of min-duration for cmb observations only.
        Clears up leftover short observations after splitting.
    max_pass : int
        Maximum number of attempts
    plan_moves : Dict[str, Any]
        Config dict for PlanMoves pass
    simplify_moves : Dict[str, Any]
        Config dict for SimplifyMoves pass

    """
    policy_config: Dict[str, Any]
    min_duration: float = 1 * u.minute
    min_cmb_duration: float = 10 * u.minute
    disable_hwp: bool = False
    max_pass: int = 3
    plan_moves: Dict[str, Any] = field(default_factory=dict)
    simplify_moves: Dict[str, Any] = field(default_factory=dict)

    def divide_cmb_scans(self, block, max_dt=dt.timedelta(minutes=60), min_dt=dt.timedelta(minutes=15)):
        duration = block.duration

        # if the block is small enough, return it as is
        if duration <= (max_dt + min_dt):
            return [block]

        n_blocks = duration // max_dt
        remainder = duration % max_dt

        # split if 1 block with remainder > min duration
        if n_blocks == 1:
            return core.block_split(block, block.t0 + max_dt)

        blocks = []
        # calculate the offset for splitting
        offset = (remainder + max_dt) / 2 if remainder.total_seconds() > 0 else max_dt

        split_blocks = core.block_split(block, block.t0 + offset)
        blocks.append(split_blocks[0])

        # split the remaining block into chunks of max duration
        for i in range(n_blocks - 1):
            split_blocks = core.block_split(split_blocks[-1], split_blocks[-1].t0 + max_dt)
            blocks.append(split_blocks[0])

        # add the remaining part
        if remainder.total_seconds() > 0:
            split_blocks = core.block_split(split_blocks[-1], split_blocks[-1].t0 + offset)
            blocks.append(split_blocks[0])

        return blocks

    def merge_cmb_blocks(self, seq, max_dt=dt.timedelta(minutes=60), min_dt=dt.timedelta(minutes=15)):
        for i in range(1, len(seq)):
            current, previous = seq[i], seq[i-1]
            # skip previously merged blocks
            if current is None or previous is None:
                continue
            # don't merge blocks that are too far apart in time
            time_gap = (current.t0 - previous.t1).total_seconds()
            combined_duration = (current.duration + previous.duration).total_seconds()
            max_combined_duration = (max_dt + min_dt).total_seconds()
            # if blocks were split from same block and are close in time
            if current.tag == previous.tag and time_gap <= min_dt.total_seconds():
                # don't merge blocks that are longer than the max length
                if combined_duration <= max_combined_duration:
                    seq[i-1] = previous.extend_right(current.duration)
                    seq[i] = None
                    # add some or all of time gaps (likely from det_setup)
                    if time_gap > 0:
                        if (seq[i-1].duration.total_seconds() + time_gap) <= max_combined_duration:
                            seq[i-1] = seq[i-1].extend_right(dt.timedelta(seconds=time_gap))
                        else:
                            seq[i-1] = seq[i-1].extend_right(dt.timedelta(seconds=(max_combined_duration - seq[i-1].duration.total_seconds())))
        return seq

    def apply(self, seq, t0, t1, state, operations):
        init_state = state
        seq_prev_ = None
        seq_ = seq
        for i in range(self.max_pass):
            logger.info(f"================ pass {i+1} ================")
            seq_ = self.round_trip(seq_, t0, t1, init_state, operations)
            if seq_ == seq_prev_:
                logger.info(f"round_trip: converged in pass {i+1}, lowering...")
                break
            seq_prev_ = seq_
        else:
            logger.warning(f"round_trip: ir did not converge after {self.max_pass} passes")

        logger.info(f"================ lowering ================")

        ir = self.lower(seq_, t0, t1, init_state, operations)
        assert ir[-1].t1 <= t1, "Going beyond our schedule limit, something is wrong!"

        logger.info(f"================ solve moves ================")
        logger.info("step 1: solve sun-safe moves")
        ir = PlanMoves(**self.plan_moves).apply(ir, t1)

        logger.info("step 2: simplify moves")
        ir = SimplifyMoves(**self.simplify_moves).apply(ir)

        # in full generality we should do some round-trips to make sure
        # the moves are still valid when we include the time costs of
        # moves. Here I'm working under the assumption that the moves
        # are very small and the time cost is negligible.

        # now we do lowering further into full ops
        logger.info(f"================ lowering (ops) ================")
        ir_ops, out_state = self.lower_ops(ir, init_state)
        logger.info(u.pformat(ir_ops))

        logger.info(f"================ done ================")

        return ir_ops, out_state

    def lower(self, seq, t0, t1, state, operations):
        ir = []

        # -----------------------------------------------------------------
        # 1. pre-session operations
        # -----------------------------------------------------------------
        logger.info("step 1: planning pre-session ops")

        ops = [op for op in operations if op['sched_mode'] == SchedMode.PreSession]
        state, _, _ = self._apply_ops(state, ops)

        # assume pre-session doesn't change az_now and el_now of init state
        # i.e., assume pre-session doesn't involve any movement, if needed,
        # movement can be performed outside schedule and propogate through
        # policy.init_state()
        ir += [
            IR(name='pre_session', subtype=IRMode.PreSession,
               t0=t0, t1=state.curr_time, operations=ops,
               az=state.az_now, alt=state.el_now)
        ]

        post_init_state = state
        post_init_ir = ir.copy()

        logger.debug(f"post-init ir: {u.pformat(ir)}")
        logger.debug(f"post-init state: {state}")

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
        logger.info("step 2: planning calibration scans")

        # compile the blocks to plan
        cal_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cal', seq))
        cmb_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cmb', seq))

        # compile calibration operations
        pre_ops  = [op for op in operations if op['sched_mode'] == SchedMode.PreCal]
        in_ops   = [op for op in operations if op['sched_mode'] == SchedMode.InCal]
        post_ops = [op for op in operations if op['sched_mode'] == SchedMode.PostCal]
        cal_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        # compile cmb operations (also needed for state propagation)
        pre_ops  = [op for op in operations if op['sched_mode'] == SchedMode.PreObs]
        in_ops   = [op for op in operations if op['sched_mode'] == SchedMode.InObs]
        post_ops = [op for op in operations if op['sched_mode'] == SchedMode.PostObs]
        cmb_ops = { 'pre_ops': pre_ops, 'in_ops': in_ops, 'post_ops': post_ops }

        logger.debug(f"cal_ops to plan: {cal_ops}")
        logger.debug(f"pre-cal state: {state}")

        cal_ref_state = state

        cal_irs = []
        # need to loop over all blocks to get the state right
        for block in seq:
            # cmb scans are just used to get the state right, no operations
            # are actually added to our list
            if block.subtype == 'cmb':
                constraint = core.Block(t0=state.curr_time, t1=block.t1)
                state, _ = self._plan_block_operations(state, block, constraint, **cmb_ops)
            elif block.subtype == 'cal':
                logger.info(f"-> planning cal block: {block}")
                logger.debug(f"--> pre-cal state: {state}")
                # what's our constraint? cal takes higher priority so we ignore causal constraint
                # from cmb scans, but we don't extend into previous cal block as they have equal
                # priority, hence our constraint start from when last cal finishes and till the 
                # end of schedule
                constraint = core.Block(t0=cal_ref_state.curr_time, t1=block.t1)

                # start at the beginning of our constraint
                state = state.replace(curr_time=constraint.t0)
                state, block_ops = self._plan_block_operations(state, block, constraint, **cal_ops)

                logger.debug(f"--> post-block ops: {u.pformat(block_ops)}")
                logger.debug(f"--> post-block state: {u.pformat(state)}")

                # update our reference state
                cal_ref_state = state
                if len(block_ops) > 0:
                    cal_irs += [block_ops]
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        logger.debug(f"final state after cal planning: {u.pformat(state)}")

        # -----------------------------------------------------------------
        # 3. cmb scans
        #
        # Note: for cmb scans, we will avoid overlapping with calibrations;
        # this means we will tend to overwrite into cmb blocks more often
        # -----------------------------------------------------------------
        logger.info("step 3: planning cmb ops")

        # restart from post-init
        state = post_init_state

        # calibration operations take precedence: now we know how long these cal
        # operations will take, we can remove overlapping regions from our cmb scans

        # avoid cmb operations from colliding with calibration operations,
        # so we identify non-overlapping intervals here. This will be used as constraint.
        ir_blocks = []
        for cal_ir in cal_irs:
            ir_blocks += [core.Block(t0=cal_ir[0].t0, t1=cal_ir[-1].t1)]
        full_interval = core.Block(t0=t0, t1=t1)
        non_overlaps = core.seq_remove_overlap(full_interval, ir_blocks)  # as constraint

        cmb_blocks = core.seq_remove_overlap(cmb_blocks, ir_blocks)
        cmb_blocks = self.merge_cmb_blocks(cmb_blocks, dt.timedelta(seconds=self.policy_config.max_cmb_scan_duration))
        cmb_blocks = core.seq_flatten(ru.MinDuration(self.min_cmb_duration)(cmb_blocks))

        # re-merge all blocks
        all_blocks = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        # done with previously planned operation seqs
        # re-plan from the end of initialization.
        state = post_init_state
        ir = post_init_ir

        for block in all_blocks:
            if state.curr_time >= block.t1:
                logger.info(f"--> skipping block {block.name} because it's already past")
                continue

            logger.debug(f"-> planning block ({block.subtype}): {block.name}: {block.t0} - {block.t1}")
            logger.debug(f"--> pre-block state: {u.pformat(state)}")

            constraint = core.Block(t0=state.curr_time, t1=block.t1)
            logger.debug(f"--> causal constraint: {constraint.t0} to {constraint.t1}")

            if block.subtype == 'cmb':
                # this should always be possible
                non_overlap = [b for b in non_overlaps if b.t0 <= block.t0 and block.t1 <= b.t1]
                assert len(non_overlap) == 1, f"unexpected non-overlapping intervals: {non_overlap=}"
                # cmb blocks are subject to additional constraint to not overlap with
                # calibrations
                non_overlap = non_overlap[0]
                logger.debug(f"--> operational constraint: {non_overlap.t0} to {non_overlap.t1}")
                constraint = core.block_intersect(constraint, non_overlap)
                ops_ = cmb_ops
                if self.policy_config.max_cmb_scan_duration is not None:
                    blocks = self.divide_cmb_scans(block, dt.timedelta(seconds=self.policy_config.max_cmb_scan_duration))
                else:
                    blocks = [block]
            elif block.subtype == 'cal':
                ops_ = cal_ops
                blocks = [block]
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")
            logger.debug(f"--> final constraint: {constraint.t0} to {constraint.t1}")

            for b in blocks:
                state, block_ops = self._plan_block_operations(state, b, constraint, **ops_)
                logger.debug(f"--> post-block state: {state}")
                ir += [block_ops]

        logger.debug(f"post-planning state: {state}")

        # -----------------------------------------------------------------
        # 4. post-session operations
        # -----------------------------------------------------------------
        logger.info("step 4: planning post-session ops")

        ops = [op for op in operations if op['sched_mode'] == SchedMode.PostSession]
        state, post_dur, _ = self._apply_ops(state, ops)

        if len(ops) > 0:
            # find an alt, az that is sun-safe for the entire duration of the schedule.
            if not self.plan_moves['stow_position']:
                az = 180
                alt = 60
                # add a buffer to start and end to be safe
                t_start = t0 - dt.timedelta(seconds=300)
                t_end = t1 + dt.timedelta(seconds=300)
                az, alt, _, _ = get_parking(t_start, t_end, alt, self.plan_moves['sun_policy'])
                logger.info(f"found sun safe stow position at ({az}, {alt})")
            # allow for overriding of stow position from config
            else:
                az = self.plan_moves['stow_position']['az_stow']
                alt = self.plan_moves['stow_position']['el_stow']
                logger.info(f"using specified stow position at ({az}, {alt})")
        else:
            az = all_blocks[-1].az
            alt = all_blocks[-1].alt

        ir += [
            IR(name='post_session', subtype=IRMode.PostSession,
            t0=state.curr_time-dt.timedelta(seconds=post_dur),
            t1=state.curr_time, operations=ops,
            az=az,
            alt=alt)
        ]

        logger.debug(f"post-session state: {state}")

        return ir

    def round_trip(self, seq, t0, t1, state, operations):
        ir = self.lower(seq, t0, t1, state, operations)
        ir_ = core.seq_flatten(ir)

        for i in range(1, len(ir_)):
            block0 = ir_[i-1].block
            block1 = ir_[i].block
            # if one block starts immediately after another and they aren't gap blocks
            if ir_[i-1].t1 == ir_[i].t0 and block0 is not None and block1 is not None:
                safet = get_traj_ok_time(
                    block0.az, block1.az,
                    block0.alt, block1.alt,
                    block0.t1, self.plan_moves['sun_policy']
                )

                shift = 10
                t0 = ir_[i-1].t1

                # is the move sun-safe
                while safet <= block1.t0:
                    # if we are currently in a obs block
                    if ir_[i].subtype == IRMode.InBlock:
                    # don't need to update anything else as we will lift the
                    # obs blocks immediately afterwards
                        ir_[i] = ir_[i].replace(
                                block=block1.shrink_left(dt.timedelta(seconds=shift))
                            )
                    # if current IR is a pre-obs block we can shrink the next
                    # observation block
                    elif ir_[i].subtype == IRMode.PreBlock and i != len(ir_)-1:
                        if ir_[i+1].subtype == IRMode.InBlock:
                            next_block = ir_[i+1].block
                            ir_[i+1] = ir_[i+1].replace(
                                block=next_block.shrink_left(dt.timedelta(seconds=shift))
                            )

                    # update obs start time
                    t0 += dt.timedelta(seconds=shift)
                    safet = get_traj_ok_time(
                        block0.az, block1.az,
                        block0.alt, block1.alt,
                        t0, self.plan_moves['sun_policy']
                    )

        seq = self.lift(ir_)

        # opportunity to do some correction:
        # if our post session is running longer than our constraint, trim the sequence
        if len(list(filter(lambda o: o['sched_mode'] == SchedMode.PostSession, operations))) > 0:
            # if we have post session, it will guarentee to be the last
            session_end = ir[-1].t1
            if session_end > t1:
                # if we are running late, truncate our blocks to make up the time
                logger.info("not enough time for post-session operations, trimming...")
                seq_t1 = seq[-1].t1
                seq = core.seq_flatten(core.seq_trim(seq, t0, seq_t1-(session_end-t1)))
        return seq

    def lift(self, ir):
        return core.seq_sort(core.seq_map(lambda b: b.block if b.subtype == IRMode.InBlock else None, ir), flatten=True)

    def lower_ops(self, irs, state):
        # `lower` generates a basic plan, here we work with ir to resolve 
        # all operations within each blocks
        def resolve_block(state, ir):
            if isinstance(ir, WaitUntil):
                op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Aux, 't1': ir.t1}]
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            elif isinstance(ir, MoveTo):
                op_cfgs = [{'name': 'move_to', 'sched_mode': IRMode.Aux, 'az': ir.az, 'el': ir.alt,
                'min_el': self.policy_config.min_hwp_el, 'force': True}]  # aux move_to should be enforced
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            elif ir.subtype in [IRMode.PreSession, IRMode.PostSession]:
                state, _, op_blocks = self._apply_ops(state, ir.operations, az=ir.az, alt=ir.alt)
            elif ir.subtype in [IRMode.PreBlock, IRMode.InBlock, IRMode.PostBlock]:
                op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Aux, 't1': ir.t0}]
                state, _, op_blocks_wait = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
                state, _, op_blocks_cmd = self._apply_ops(state, ir.operations, block=ir.block)
                op_blocks = op_blocks_wait + op_blocks_cmd
            elif ir.subtype == IRMode.Gap:
                op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Gap, 't1': ir.t1}]
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            else:
                raise ValueError(f"unexpected block type: {ir}")
            return state, op_blocks

        ir_lowered = []
        for ir in irs:
            state, op_blocks = resolve_block(state, ir)
            ir_lowered += op_blocks
        return ir_lowered, state

    def _apply_ops(self, state, op_cfgs, block=None, az=None, alt=None):
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
        if (az is None or alt is None) and (block is not None):
            az, alt = block.az, block.alt

        for op_cfg_ in op_cfgs:
            op_cfg = op_cfg_.copy()

            # sanity check
            for k in ['name', 'sched_mode']:
                assert k in op_cfg, f"operation config must have a '{k}' key"

            # pop some non-operation kwargs
            op_name = op_cfg.pop('name')
            sched_mode = op_cfg.pop('sched_mode')

            # not needed now -> needed only during lowering
            op_cfg.pop('indent', None)
            op_cfg.pop('divider', None)

            # add block to the operation config if provided
            block_cfg = {'block': block} if block is not None else {}

            op_cfg = {**op_cfg, **block_cfg}  # make copy

            # apply operation
            t_start = state.curr_time
            op = cmd.make_op(op_name, **op_cfg)
            state, dur, _ = op(state)

            duration += dur
            state = state.increment_time(dt.timedelta(seconds=dur))

            op_blocks += [IR(
                name=op_name,
                subtype=sched_mode,
                t0=t_start,
                t1=state.curr_time,
                az=az,
                alt=alt,
                block=block,
                operations=[op_cfg_]
            )]

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
        List[IR]
            Sequence of operations planned for the block.

        """

        # fast forward to within the constrained time block
        state = state.replace(curr_time=max(constraint.t0, state.curr_time))

        shift = 10
        safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, state.curr_time, self.plan_moves['sun_policy'])
        while safet <= state.curr_time:
            state = state.replace(curr_time=state.curr_time + dt.timedelta(seconds=shift))
            safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, state.curr_time, self.plan_moves['sun_policy'])

        # if we already pass the block or our constraint, nothing to do
        if state.curr_time >= block.t1 or state.curr_time >= constraint.t1:
            logger.debug(f"--> skipping block {block.name} because it's already past")
            return state, []

        initial_state = state
        logger.debug(f"--> with constraint: planning {block.name} from {state.curr_time} to {block.t1}")

        op_seq = []

        # +++++++++++++++++++++
        # pre-block operations
        # +++++++++++++++++++++
        logger.debug(f"--> planning pre-block operations")

        # did we already pass the block?
        t_start = state.curr_time
        if t_start >= block.t1:
            logger.debug(f"---> skipping block {block.name} because it's already past")
            return initial_state, []
        state, pre_dur, _ = self._apply_ops(state, pre_ops, block=block)

        logger.debug(f"---> pre-block ops duration: {pre_dur} seconds")
        logger.debug(f"---> pre-block curr state: {u.pformat(state)}")

        # what time are we starting?
        # -> start from t_start or block.t0-duration, whichever is later
        # -> overwrite block if we extended into the block
        # -> if we extended past the block, skip operation

        # did we extend into the block?
        if state.curr_time > block.t0:
            logger.debug(f"---> curr_time extended into block {block.name}")
            # did we extend past entire block?
            if state.curr_time < block.t1:
                logger.debug(f"---> curr_time did not extend past block {block.name}")
                block = block.trim_left_to(state.curr_time)
                logger.debug(f"---> trimmed block: {block}")
                pre_block_name = "pre_block (into)"
            else:
                return initial_state, []
        else:
            logger.debug(f"---> gap is large enough for pre-block operations")
            # if there is more time than we need, let's wait to start later: 
            # state = initial_state.replace(curr_time=block.t0-dt.timedelta(seconds=duration))
            state = state.replace(curr_time=block.t0)
            pre_block_name = "pre_block"

        logger.debug(f"--> post pre-block state: {u.pformat(state)}")
        logger.debug(f"--> post pre-block op_seq: {u.pformat(op_seq)}")

        # +++++++++++++++++++
        # in-block operations
        # +++++++++++++++++++

        logger.debug(f"--> planning in-block operations from {state.curr_time} to {block.t1}")
        logger.debug(f"--> pre-planning state: {u.pformat(state)}")

        state, in_dur, _ = self._apply_ops(state, in_ops, block=block)

        logger.debug(f"---> in-block ops duration: {in_dur} seconds")
        logger.debug(f"---> in-block curr state: {u.pformat(state)}")

        # sanity check: if fail, it means post-cal operations are 
        # mixed into in-cal operations
        assert state.curr_time <= block.t1, \
            "in-block operations are probably mixed with post-cal operations"

        # advance to the end of the block
        state = state.replace(curr_time=block.t1)

        logger.debug(f"---> post in-block state: {u.pformat(state)}")

        # +++++++++++++++++++++
        # post-block operations
        # +++++++++++++++++++++
        t_start = state.curr_time

        state, post_dur, _ = self._apply_ops(state, post_ops, block=block)

        logger.debug(f"---> post-block ops duration: {post_dur} seconds")
        logger.debug(f"---> post-block curr state: {u.pformat(state)}")

        # have we extended past our constraint?
        post_block_name = "post_block"
        if state.curr_time > constraint.t1:
            logger.debug(f"---> post-block ops extended past constraint")
            # shrink our block to make space for post-block operation and
            # revert to an old state before retrying
            block = block.shrink_right(state.curr_time - constraint.t1)
            # if we passed the block, there is not enough time to do anything
            # -> revert to initial state
            if block is None:
                logger.info(f"--> skipping {block=} because post-block op couldn't fit inside constraint")
                return initial_state, []
            post_block_name = "post_block (into)"
            state = state.replace(curr_time=constraint.t1)

        # block has been trimmed properly, so we can just do this
        if len(pre_ops) > 0:
            op_seq += [
                IR(name=pre_block_name,
                subtype=IRMode.PreBlock,
                t0=block.t0-dt.timedelta(seconds=pre_dur),
                t1=block.t0,
                az=block.az,
                alt=block.alt,
                block=block,
                operations=pre_ops),
            ]
        if len(in_ops) > 0:
            op_seq += [
                IR(name=block.name,
                subtype=IRMode.InBlock,
                t0=block.t0,
                t1=block.t1,
                az=block.az,
                alt=block.alt,
                block=block,
                operations=in_ops),
            ]

        if len(post_ops) > 0:
            op_seq += [
                IR(name=post_block_name,
                subtype=IRMode.PostBlock,
                t0=block.t1,
                t1=block.t1+dt.timedelta(seconds=post_dur),
                az=block.az,
                alt=block.alt,
                block=block,
                operations=post_ops)
            ]

        return state, op_seq

@dataclass(frozen=True)
class PlanMoves:
    """solve moves to make seq possible"""
    sun_policy: Dict[str, Any]
    stow_position: Dict[str, Any]
    az_step: float = 1
    az_limits: Tuple[float, float] = (-90, 450)

    def apply(self, seq, t_end):
        """take a list of IR from BuildOp as input to solve for optimal sun-safe moves"""

        seq = core.seq_sort(seq, flatten=True)

        def get_safe_gaps(block0, block1):
            """Returns a list with 0, 1, or 3 Gap blocks.  The Gap blocks will be
            at sunsafe positions for their duration, and be safely
            reachable in the sequence block0 -> gaps -> block1.

            The az and alt specified for each gap will be sun-safe for their duration.

            """
            if (block0.t1 >= block1.t0):
                return []
            # Check the move
            t1 = get_traj_ok_time(block0.az, block1.az, block0.alt, block1.alt,
                                  block0.t1, self.sun_policy)
            if t1 >= block1.t0:
                return [IR(name='gap', subtype=IRMode.Gap, t0=block0.t1, t1=block1.t0,
                           az=block1.az, alt=block1.alt)]

            az_parking, alt_parking, t0_parking, t1_parking = get_parking(block0.t1, block1.t0,
                                                                          block0.alt, self.sun_policy)

            def check_parking(t0_parking, t1_parking, alt_parking):
                if get_traj_ok_time(az_parking, az_parking, alt_parking, alt_parking,
                                    t0_parking, self.sun_policy) < t1_parking:
                    raise ValueError("Sun-safe parking spot not found.")

            # You might need to rush away from final position...
            move_away_by = get_traj_ok_time(
                block0.az, az_parking, block0.alt, alt_parking, block0.t1, self.sun_policy)

            if move_away_by < t0_parking:
                if move_away_by < block0.t1:
                    raise ValueError("Sun-safe parking spot not accessible from prior scan.")
                else:
                    t0_parking = move_away_by + (move_away_by - block0.t1) / 2

            # You might need to wait until the last second before going to new pos
            max_delay = 300
            shift = 10.
            while t1_parking < block1.t0 + dt.timedelta(seconds=max_delay):
                ok_until = get_traj_ok_time(
                    az_parking, block1.az, alt_parking, block1.alt, t1_parking, self.sun_policy)
                if ok_until >= block1.t0:
                    break
                t1_parking = t1_parking + dt.timedelta(seconds=shift)
            else:
                raise ValueError("Next scan not accessible from sun-safe parking spot.")

            if t1_parking > block1.t0:
                logger.warning("sun-safe parking delays move to next field by "
                               f"{(t1_parking - block1.t0).total_seconds()} seconds")

            return [IR(name='gap', subtype=IRMode.Gap, t0=block0.t1, t1=t0_parking,
                       az=az_parking, alt=alt_parking),
                    IR(name='gap', subtype=IRMode.Gap, t0=t0_parking, t1=t1_parking,
                       az=az_parking, alt=alt_parking),
                    IR(name='gap', subtype=IRMode.Gap, t0=t1_parking, t1=block1.t0,
                       az=block1.az, alt=block1.alt),
                    ]

        # go through the sequence and wrap az if falls outside limits
        seq_ = []
        for b in seq:
            if b.az < self.az_limits[0] or b.az > self.az_limits[1]:
                logger.info(f"block az ({b.az}) outside limits, unwrapping...")
                az_unwrap = find_unwrap(b.az, az_limits=self.az_limits)[0]
                logger.info(f"-> unwrapping az: {b.az} -> {az_unwrap}")
                seq_ += [b.replace(az=az_unwrap)]
            else:
                seq_ += [b]
        seq = seq_

        seq_ = [seq[0]]
        for i in range(1, len(seq)):
            gaps = get_safe_gaps(seq[i-1], seq[i])
            seq_.extend(gaps)
            seq_.append(seq[i])

        # find sun-safe parking if not stowing at end of schedule
        if seq[-1].name in 'post_session' and len(seq[-1].operations) == 0:
            block = seq[-1]
            safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, block.t1, self.sun_policy)
            # if current position is safe until end of schedule
            if safet >= t_end:
                seq_.extend([IR(name='gap', subtype=IRMode.Gap, t0=block.t1, t1=t_end,
                    az=block.az, alt=block.alt)])
            else:
                movet = block.t1 #max(safet, block.t1)
                az_parking, alt_parking, t0_parking, t1_parking = get_parking(movet, t_end, block.alt,
                                                                              self.sun_policy)

                get_safe_gaps(block, IR(name='gap', subtype=IRMode.Gap, t0=t0_parking, t1=t1_parking,
                        az=az_parking, alt=alt_parking))

                move_away_by = get_traj_ok_time(
                    block.az, az_parking, block.alt, alt_parking, movet, self.sun_policy)
                if move_away_by < t0_parking:
                    if move_away_by < movet:
                        raise ValueError("Sun-safe parking spot not accessible from prior scan.")
                    else:
                        t0_parking = move_away_by + (move_away_by - movet) / 2

                seq_.extend([IR(name='gap', subtype=IRMode.Gap, t0=block.t1, t1=movet,
                        az=block.az, alt=block.alt),
                        IR(name='gap', subtype=IRMode.Gap, t0=t0_parking, t1=t1_parking,
                        az=az_parking, alt=alt_parking)])


        # Replace gaps with Wait, Move, Wait.
        seq_, seq = [], seq_
        last_az, last_alt = None, None
        # Combine, but skipping first and last blocks, which are init/shutdown.
        for i, b in enumerate(seq):
            if b.name in ['pre_session']:
                # Pre/post-ambles, leave it alone.
                seq_ += [b]
            elif b.name == 'gap':
                # For a gap, always seek to the stated gap position.
                # But not until the gap is supposed to start.  Since
                # gaps may be used to manage Sun Avoidance, it's
                # important to be in that place for that time period.
                seq_ += [
                    WaitUntil(t1=b.t0, az=b.az, alt=b.alt),
                    MoveTo(az=b.az, alt=b.alt),
                    WaitUntil(t1=b.t1, az=b.az, alt=b.alt)]
                last_az, last_alt = b.az, b.alt
            else:
                if (last_az is None
                    or np.round(b.az - last_az, 3) != 0
                    or np.round(b.alt - last_alt, 3) != 0):
                    seq_ += [MoveTo(az=b.az, alt=b.alt)]
                    last_az, last_alt = b.az, b.alt
                else:
                    if (b.block != seq[i-1].block) & (i>0):
                        seq_ += [MoveTo(az=b.az, alt=b.alt)]
                seq_ += [b]

        return seq_


@dataclass(frozen=True)
class SimplifyMoves:
    def apply(self, ir):
        """simplify moves by removing redundant MoveTo blocks"""
        i_pass = 0
        while True:
            logger.info(f"simplify_moves: {i_pass=}")
            ir_new = self.round_trip(ir)
            #ir_new = ir
            if ir_new == ir:
                logger.info("simplify_moves: IR converged")
                return ir
            ir = ir_new
            i_pass += 1

    def round_trip(self, ir):
        def without(i):
            return ir[:i] + ir[i+1:]
        for bi in range(len(ir)-1):
            b1, b2 = ir[bi], ir[bi+1]
            if isinstance(b1, MoveTo) and isinstance(b2, MoveTo):
                # repeated moves will be replaced by the last move
                # b = MoveTo(az=b2.az, alt=b2.alt)
                return without(bi)
            elif isinstance(b1, WaitUntil) and isinstance(b2, WaitUntil):
                # repeated wait untils will be replaced by the longer wait
                if b1.t1 < b2.t1:
                    return without(bi)
                return without(bi+1)
            elif (isinstance(b1, IR) and b1.subtype == IRMode.Gap) and isinstance(b2, WaitUntil):
                # gap followed by wait until will be replaced by the wait until
                return without(bi)
            # remove redundant move->wait->move if they are all at the same az/alt
            for bi in range(len(ir) - 3):
                if (
                    isinstance(ir[bi], MoveTo)
                    and isinstance(ir[bi + 2], MoveTo)
                    and ir[bi] == ir[bi + 2]
                    and isinstance(ir[bi + 1], WaitUntil)
                ):
                    return ir[:bi + 1] + ir[bi + 3:]

        return ir


def find_unwrap(az, az_limits=[-90, 450]) -> List[float]:
    az = (az - az_limits[0]) % 360 + az_limits[0]  # min az becomes az_limits[0]
    az_unwraps = list(np.arange(az, az_limits[1], 360))
    return az_unwraps

def az_ranges_intersect(
    r1: List[Tuple[float, float]],
    r2: List[Tuple[float, float]],
    *,
    az_limits: Tuple[float, float],
    az_step: float
) -> List[Tuple[float, float]]:
    az_full = np.arange(az_limits[0], az_limits[1]+az_step, az_step)
    mask1 = np.zeros_like(az_full, dtype=bool)
    mask2 = np.zeros_like(az_full, dtype=bool)
    for s in r1:
        mask1 = np.logical_or(mask1, (az_full >= s[0]) * (az_full <= s[1]))
    for s in r2:
        mask2 = np.logical_or(mask2, (az_full >= s[0]) * (az_full <= s[1]))
    sints_both = [(az_full[s[0]], az_full[s[1]-1]) for s in u.mask2ranges(mask1*mask2)]
    return sints_both

def az_distance(az1: float, az2: float) -> float:
    return abs(az1 - az2)
    # return abs((az1 - az2 + 180) % 360 - 180)

def az_ranges_contain(ranges: List[Tuple[float,float]], az: float) -> bool:
    for r in ranges:
        if r[0] <= az <= r[1]:
            return True
    return False

def az_ranges_cover(ranges: List[Tuple[float,float]], range_: Tuple[float, float]) -> bool:
    for r in ranges:
        if r[0] <= range_[0] and r[1] >= range_[1]:
            return True
    return False
