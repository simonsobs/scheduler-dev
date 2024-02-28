"""A production-level implementation of the SAT policy

"""
import yaml
import os.path as op
from dataclasses import dataclass, field
import datetime as dt
from typing import List, Union, Optional, Dict
import jax.tree_util as tu
import numpy as np
from collections import OrderedDict

from .. import config as cfg, core, source as src, rules as ru, commands as cmd, instrument as inst

# ==================
# useful commands
# ==================
preamble = [
    "from nextline import disable_trace",
    "",
    "import time",
    "import datetime",
    "",
    "with disable_trace():",
    "    import numpy as np",
    "    import sorunlib as run",
    "    run.initialize()",
    "    from ocs.ocs_client import OCSClient",
    "",
    "UTC = datetime.timezone.utc",
    "acu = run.CLIENTS['acu']",
    "pysmurfs = run.CLIENTS['smurf']",
    "",
]

wrap_up = [
    "# go home",
    "run.acu.move_to(az=180, el=50)",
    "",
    "time.sleep(1)"
]

ufm_relock = [
    "############# Daily Relock ######################",
    "for smurf in pysmurfs:",
    "    smurf.zero_biases.start()",
    "for smurf in pysmurfs:",
    "    smurf.zero_biases.wait()",
    "",
    "time.sleep(60)",
    "run.smurf.take_noise(concurrent=True, tag='oper,take_noise,res_check')",
    "",
    "run.smurf.uxm_relock(concurrent=True)",
    "#################################################", 
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
    merge_order : list
        a list of queries, specifies the order of merging, e.g., ['baseline', 'calibration']
        indicates that baseline blocks takes precedence over calibration blocks in case of
        overlap
    time_costs : dict
        a dict of time costs, specifies the time cost of various operations, e.g., 'det_setup'
        specifies the time cost of detector setup
    ufm_relock : bool
        whether to relock UFM before the start of the sequence
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
    # internal use only
    checkpoints : dict
        a dict of checkpoints, with keys being checkpoint names and values being blocks 
    """
    blocks: dict
    rules: Dict[str, core.Rule]
    geometries: List[dict]
    cal_targets: List[tuple]
    merge_order: List[str]
    time_costs: dict[str, float]
    ufm_relock: bool
    scan_tag: Optional[str] = None
    az_speed: float = 0.8 # deg / s
    az_accel: float = 0.25 # deg / s^2
    apply_boresight_rot: bool = False
    allow_partial: bool = False
    wafer_sets: dict[str, str] = field(default_factory=dict)
    checkpoints: dict[str, core.BlocksTree] = field(default_factory=OrderedDict)
    
    def save_checkpoint(self, name, blocks):
        self.checkpoints[name] = blocks

    @classmethod
    def from_config(cls, config: Union[dict, str]):
        """populate policy object from a yaml config file or a string yaml
        config or a dict"""
        if isinstance(config, str):
            loader = cfg.get_loader()
            if op.isfile(config):
                with open(config, "r") as f:
                    config = yaml.load(f.read(), Loader=loader)
            else:
                config = yaml.load(config, Loader=loader)

        # now we can construct the policy
        return cls(**config)

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        def construct_seq(loader_cfg):
            if loader_cfg['type'] == 'source':
                return src.source_gen_seq(loader_cfg['name'], t0, t1)
            elif loader_cfg['type'] == 'toast':
                return inst.parse_sequence_from_toast(loader_cfg['file'])
            else:
                raise ValueError(f"unknown sequence type: {loader_cfg['type']}")
        blocks = tu.tree_map(construct_seq, self.blocks, 
                             is_leaf=lambda x: isinstance(x, dict) and 'type' in x)
        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            source = cal_target[0]
            if source not in blocks['calibration']:
                blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)
        return core.seq_trim(blocks, t0, t1)

    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        # save the original blocks
        self.save_checkpoint('original', blocks)

        # sun avoidance
        if 'sun-avoidance' in self.rules:
            rule = ru.make_rule('sun-avoidance', **self.rules['sun-avoidance'])
            blocks = rule(blocks)
            self.save_checkpoint('sun-avoidance', blocks)
        
        # plan source scans
        cal_blocks = {}
        for cal_target in self.cal_targets:
            if len(cal_target) == 4:
                source, array_query, el_bore, tagname = cal_target
                boresight_rot = None
            elif len(cal_target) == 5:
                source, array_query, el_bore, boresight_rot, tagname = cal_target
            else:
                raise ValueError("cal_target has an unrecognized format")

            assert source in blocks['calibration'], f"source {source} not found in sequence"

            # translation: allow array_query to look up from 
            # wafer_set definitions
            if array_query in self.wafer_sets:
                array_query = self.wafer_sets[array_query]

            # build geometry information
            array_info = inst.array_info_from_query(self.geometries, array_query)
            rule = ru.MakeCESourceScan(
                array_info=array_info, 
                el_bore=el_bore, 
                drift=True,
                boresight_rot=boresight_rot,
                allow_partial=self.allow_partial,
            )
            if source not in cal_blocks: cal_blocks[source] = []
            _blocks = rule(blocks['calibration'][source])
            cal_blocks[source].append(
                core.seq_map(lambda block: block.replace(tag=f"{block.tag},{tagname}"), _blocks)
            )

        # can we simply merge these blocks for each source?
        for source in cal_blocks:
            # if not, let's alternate between targets
            if core.seq_has_overlap(cal_blocks[source]):
                # one target per source block (e.g. per transit)
                rules = []
                for cal_target in self.cal_targets:
                    if len(cal_target) == 4:
                        source_, array_query, el_bore, tagname = cal_target
                        boresight_rot = None
                    elif len(cal_target) == 5:
                        source_, array_query, el_bore, boresight_rot, tagname = cal_target
                    else:
                        raise ValueError("cal_target has an unrecognized format")
                    if source_ != source: continue
                    # translation: allow array_query to look up from 
                    # wafer_set definitions
                    if array_query in self.wafer_sets:
                        array_query = self.wafer_sets[array_query]
                    rules.append(
                        (
                            tagname,
                            ru.MakeCESourceScan(
                                array_info=inst.array_info_from_query(self.geometries, array_query),
                                el_bore=el_bore,
                                drift=True,
                                boresight_rot=boresight_rot,
                                allow_partial=self.allow_partial,
                            ),
                        )
                    )

                new_blocks = []
                rule_i = 0
                for block in blocks['calibration'][source]:
                    if block is None: continue
                    tagname, rule = rules[rule_i]
                    block = rule(block)
                    if block is None: continue
                    new_blocks.append(block.replace(tag=f"{block.tag},{tagname}"))
                    rule_i = (rule_i + 1) % len(rules)  # alternating between rules
                cal_blocks[source] = new_blocks
            else:
                cal_blocks[source] = core.seq_flatten(cal_blocks[source])
        
        # store the result back to calibration 
        # (not in-place so previous checkpoints are not affected)
        blocks = blocks.copy()
        blocks['calibration'] = cal_blocks
        self.save_checkpoint('add-calibration', blocks)

        # az range fix
        if 'az-range' in self.rules:
            rule = ru.make_rule('az-range', **self.rules['az-range'])
            blocks = rule(blocks)
            self.save_checkpoint('az-range', blocks)

        # add proper subtypes
        blocks['calibration'] = core.seq_map(lambda block: block.replace(subtype="cal"), blocks['calibration'])
        blocks['baseline']['cmb'] = core.seq_map(lambda block: block.replace(subtype="cmb", tag=f"{block.az:.0f}-{block.az+block.throw:.0f}"), blocks['baseline']['cmb'])

        # add scan tag if supplied
        if self.scan_tag is not None:
            blocks['baseline'] = core.seq_map(lambda block: block.replace(tag=f"{block.tag},{self.scan_tag}"), blocks['baseline'])

        #########
        # merge #
        #########

        seq = None
        for query in self.merge_order[::-1]:
            match, _ = core.seq_partition_with_query(query, blocks)
            if seq is None: 
                seq = match
                continue
            else:
                # match takes precedence
                seq = core.seq_merge(seq, match, flatten=True)

        self.save_checkpoint('merge', seq)

        # duration cut
        if 'min-duration' in self.rules:
            rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            seq = rule(seq)

        # save the result
        self.save_checkpoint('final', seq)
        return core.seq_sort(seq)

    def seq2cmd(self, seq: core.Blocks, t0: dt.datetime, t1: dt.datetime):
        time_cost = 0  # secs
        commands = []

        commands += preamble

        if self.ufm_relock:
            commands += ufm_relock
            time_cost += self.time_costs['ufm_relock']

        # set az speed and accel
        commands += [
            "",
            f"run.acu.set_scan_params({self.az_speed}, {self.az_accel})",
            "",
        ] 
        
        # start to build scans
        assert core.seq_is_sorted(seq), "seq must be sorted"

        t_cur = t0 + dt.timedelta(seconds=time_cost)

        is_det_setup = False
        cur_boresight_angle = None
        for block in seq:
            # det setup
            if t_cur + dt.timedelta(seconds=self.time_costs['det_setup']) > block.t1:
                commands += [
                    "\"\"\"",
                    f"Note: {block} skipped due to insufficient time",
                    "\"\"\"",
                ]
                continue
            else:
                if block.subtype == 'cmb':
                    if not is_det_setup:
                        t_start = block.t0 - dt.timedelta(seconds=self.time_costs['det_setup'])
                        f"print('Waiting until {t_start} to start detector setup')",
                        commands += [
                            "",
                            f"run.wait_until('{t_start.isoformat()}')",
                            "###################Detector Setup######################",
                            
                            f"run.acu.move_to(az={round(np.mod(block.az,360),3)}, el={round(block.alt,3)})",
                            "run.smurf.take_bgmap(concurrent=True)",
                            "run.smurf.iv_curve(concurrent=True)",
                            "for smurf in pysmurfs:",
                            "    smurf.bias_dets.start(rfrac=0.5, kwargs=dict(bias_groups=[0,1,2,3,4,5,6,7,8,9,10,11]))",
                            "time.sleep(5*60)",
                            "run.smurf.bias_step(concurrent=True)",
                        
                            "#################### Detector Setup Over ####################",
                            "",
                            "",
                        ]
                        is_det_setup = True

                    commands += [
                        "",
                        "#~~~~~~~~~~~~~~~~~~~~~~~",
                        f"run.wait_until('{block.t0.isoformat()}')"
                    ]

                    if self.apply_boresight_rot and block.boresight_angle is not None and block.boresight_rot != cur_boresight_angle:
                        commands += [
                            f"run.acu.set_boresight({block.boresight_angle})",
                        ]
                        cur_boresight_angle = block.boresight_rot

                    commands += [
                        f"run.acu.move_to(az={round(block.az,3)}, el={round(block.alt,3)})",
                         "run.smurf.bias_step(concurrent=True)",
                         "run.seq.scan(",
                        f"        description='{block.name}',",
                        f"        stop_time='{block.t1.isoformat()}',", 
                        f"        width={round(block.throw,3)}, az_drift=0,",
                        f"        subtype='cmb', tag='{block.tag}',",
                         ")",
                         "#~~~~~~~~~~~~~~~~~~~~~~~",
                         "",
                    ]
                if block.subtype == 'cal':
                    t_start = block.t0 - dt.timedelta(seconds=self.time_costs['det_setup'])
                    
                    # setup detectors
                    commands += [
                        "",
                        "#################### Detector Setup #########################",
                        f"print('Waiting until {t_start} to start detector setup')",
                        f"run.wait_until('{t_start.isoformat()}')",
                    ]

                    if self.apply_boresight_rot and block.boresight_angle is not None and block.boresight_rot != cur_boresight_angle:
                        commands += [
                            f"run.acu.set_boresight({block.boresight_angle})",
                        ]
                        cur_boresight_angle = block.boresight_rot

                    commands += [
                        f"run.acu.move_to(az={round(np.mod(block.az,360),3)}, el={round(block.alt,3)})",
                        "run.smurf.take_bgmap(concurrent=True)",
                        "run.smurf.iv_curve(concurrent=True)",
                        "for smurf in pysmurfs:",
                        "    smurf.bias_dets.start(rfrac=0.5, kwargs=dict(bias_groups=[0,1,2,3,4,5,6,7,8,9,10,11]))",
                        "time.sleep(5*60)",
                        "run.smurf.bias_step(concurrent=True)",
                        "#################### Detector Setup Over ####################",
                        "",
                    ]
                    is_det_setup = True

                    # start the scan
                    commands += [
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
                        "    print('Taking Bias Steps')",
                        "    run.smurf.bias_step(concurrent=True)",
                        "################# Scan Over #############################",
                    ] 
                t_cur = block.t1 + dt.timedelta(seconds=self.time_costs['bias_step'])

        commands += wrap_up

        return cmd.CompositeCommand(commands)