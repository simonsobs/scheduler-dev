import numpy as np
from typing import Optional
from dataclasses import dataclass
import datetime as dt

from .. import source as src, utils as u, commands as cmd
from .sat import SchedMode, SATPolicy

logger = u.init_logger(__name__)

# ----------------------------------------------------
#
#     Define satp1 specific state
#
# Note: it inherits fields from the generic sat state
# operations should start with `satp1.`
#
# ----------------------------------------------------

@dataclass(frozen=True)
class SATP1State(cmd.State):
    """
    State relevant to SATP1 operations. Inherits other fields:
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
    """
    boresight_rot_now: int = 0
    hwp_spinning: bool = False
    last_ufm_relock: Optional[dt.datetime] = None
    last_bias_step: Optional[dt.datetime] = None

# ----------------------------------------------------
#
#     Register satp1 specific operations
#
# Note: to avoid naming collisions, all satp1 specific
# operations should start with `satp1.`
#
# ----------------------------------------------------

# Although I have put everything that satp1 uses here, but some of the
# operations may be generic enough that all platforms can share, so
# I anticipate moving those out of this module soon.

@cmd.operation(name="satp1.preamble", duration=0)
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

@cmd.operation(name='satp1.ufm_relock', return_duration=True)
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

@cmd.operation(name='satp1.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, disable_hwp=False):
    if not disable_hwp and not state.hwp_spinning:
        state = state.replace(hwp_spinning=True)
        return state, 20*u.minute, [
            "HWPPrep()",
            "forward = True",
            "hwp_freq = 2.0",
            "HWPSpinUp()",
        ]
    return state, 0, ["# hwp disabled or already spinning"]

@cmd.operation(name='satp1.hwp_spin_down', return_duration=True)
def hwp_spin_down(state, disable_hwp=False):
    if not disable_hwp and state.hwp_spinning:
        state = state.replace(hwp_spinning=False)
        return state, 10*u.minute, [
            "HWPFastStop()",
            "HWPPost()",
            "hwp_freq = 0.0",
        ]
    return state, 0, ["# hwp disabled or not spinning"]

# per block operation: block will be passed in as parameter
@cmd.operation(name='satp1.det_setup', return_duration=True)
def det_setup(state, block, **hwp_kwargs):
    # only do it if boresight has changed
    duration = 0
    commands = []
    if block.alt != state.el_now:
        if state.hwp_spinning:
            # equivalent to hwp_spin_down(**hwp_kwargs)(state)
            # use make_op wrapper is more reliable in case of decorator
            # implementation change in the future
            state, d, c = cmd.make_op('satp1.hwp_spin_down', **hwp_kwargs)(state)
            commands += c
            duration += d
        commands += [
            "",
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
        state = state.replace(az_now=block.az, el_now=block.alt, last_bias_step=state.curr_time)
        duration += 60
        if not state.hwp_spinning:
            state, d, c = cmd.make_op('satp1.hwp_spin_up', **hwp_kwargs)(state)
            commands += c
            duration += d

    return state, duration, commands

@cmd.operation(name='satp1.cmb_scan', return_duration=True)
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

@cmd.operation(name='satp1.source_scan', return_duration=True)
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

@cmd.operation(name='satp1.setup_boresight', duration=0)  # TODO check duration
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    if apply_boresight_rot and state.boresight_rot_now != block.boresight_angle:
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)

    if block.alt != state.el_now:
        commands += [ f"run.acu.move_to(az={round(block.az,3)}, el={round(block.alt,3)})" ]
        state = state.replace(az_now=block.az, el_now=block.alt)
    return state, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='satp1.bias_step', return_duration=True)
def bias_step(state, min_interval=10*u.minute):
    if state.last_bias_step is None or (state.curr_time - state.last_bias_step).total_seconds() > min_interval:
        state = state.replace(last_bias_step=state.curr_time)
        return state, 60, [ "run.smurf.bias_step(concurrent=True)" ]
    else:
        return state, 0, []

@cmd.operation(name='satp1.wait_until', return_duration=True)
def wait_until(state, t1: dt.datetime):
    return state, (t1-state.curr_time).total_seconds(), [
        f"run.wait_until('{t1.isoformat()}')"
    ]

@cmd.operation(name='satp1.wrap_up', duration=0)
def wrap_up(state, az_stow, el_stow):
    state = state.replace(az_now=az_stow, el_now=el_stow)
    return state, [
        "# go home",
        f"run.acu.move_to(az={az_stow}, el={el_stow})",
        "time.sleep(1)"
    ]

@cmd.operation(name='satp1.set_scan_params', duration=0)
def set_scan_params(state, az_speed, az_accel):
    if az_speed != state.az_speed_now or az_accel != state.az_accel_now:
        state = state.replace(az_speed_now=az_speed, az_accel_now=az_accel)
        return state, [ f"run.acu.set_scan_params({az_speed}, {az_accel})"]
    return state, []

# ----------------------------------------------------
#
#         Setup satp1 specific configs
#
# ----------------------------------------------------

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

def get_operations(az_speed, az_accel, disable_hwp=False, apply_boresight_rot=True, hwp_cfg=None):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper' }
    pre_session_ops = [
        { 'name': 'satp1.preamble'        , 'sched_mode': SchedMode.PreSession, 'hwp_cfg': hwp_cfg, },
        { 'name': 'satp1.ufm_relock'      , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'satp1.set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
        { 'name': 'satp1.hwp_spin_up'     , 'sched_mode': SchedMode.PreSession, 'disable_hwp': disable_hwp, },
    ]
    post_session_ops = [
        { 'name': 'satp1.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, },
        { 'name': 'satp1.wrap_up'         , 'sched_mode': SchedMode.PostSession, 'az_stow': 180, 'el_stow': 48},
    ]
    cal_ops = [
        { 'name': 'satp1.hwp_spin_down'   , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, },
        { 'name': 'satp1.det_setup'       , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, },
        { 'name': 'satp1.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'satp1.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp},
        { 'name': 'satp1.source_scan'     , 'sched_mode': SchedMode.InCal },
        { 'name': 'satp1.bias_step'       , 'sched_mode': SchedMode.PostCal, 'indent': 4},
    ]
    cmb_ops = [
        { 'name': 'satp1.det_setup'       , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, },
        { 'name': 'satp1.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, },
        { 'name': 'satp1.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'satp1.bias_step'       , 'sched_mode': SchedMode.PreObs, },
        { 'name': 'satp1.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
        { 'name': 'satp1.bias_step'       , 'sched_mode': SchedMode.PostObs, },
    ]
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops


def get_config(
    master_file,
    az_speed,
    az_accel,
    cal_targets,
    **op_cfg
):
    blocks = get_blocks(master_file)
    geometries = get_geometry()
    operations = get_operations(az_speed, az_accel, **op_cfg)

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
#         Policy customizations
#
# ----------------------------------------------------

@dataclass
class SATP1Policy(SATPolicy):
    @classmethod
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5, cal_targets=[], **op_cfg):
        return cls(**get_config(master_file, az_speed, az_accel, cal_targets, **op_cfg))

    def add_cal_target(self, source: str, boresight: int, elevation: int, focus: str):
        self.cal_targets.append(get_cal_target(source, boresight, elevation, focus))

    def init_state(self, t0: dt.datetime) -> SATP1State:
        return SATP1State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=0,
            hwp_spinning=False,
        )