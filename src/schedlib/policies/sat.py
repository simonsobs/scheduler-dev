"""A production-level implementation of the SAT policy

"""
import yaml
import os.path as op
from dataclasses import dataclass
import datetime as dt
from typing import List 
import jax.tree_util as tu

from . import basic
from .. import config as cfg, core, source as src, rules as ru, commands as cmd, instrument as inst

geometries = {
  'full': {
    'left': {
      'ws6': {
        'center': [-10.9624, 6.46363],
        'radius': 6,
      },
      'ws5': {
        'center': [-10.9624, -6.46363],
        'radius': 6,
      },
    },
    'middle': {
      'ws1': {
        'center': [0, 12.634],
        'radius': 6,
      },
      'ws0': {
        'center': [0, 0], 
        'radius': 6,
      },
      'ws4': {
        'center': [0, -12.634],
        'radius': 6,
      },
    },
    'right': {
      'ws2': {
        'center': [10.9624, 6.46363],
        'radius': 6,
      },
      'ws3': {
        'center': [10.9624, -6.46363],
        'radius': 6,
      },
    },
  },
  'bottom' : {
    'ws4': {
        'center': [0, -12.634],
        'radius': 6,
    },
    'ws3': {
        'center': [10.9624, -6.46363],
        'radius': 6,
    },
    'ws5': {
        'center': [-10.9624, -6.46363],
        'radius': 6,
    },
  }
}

blocks = {
    'calibration': {
        'saturn': {
            'type' : 'source',
            'name' : 'saturn',
        },
    },
    'baseline': {
        'cmb': {
            'type': 'toast',
            'file': None
        }
    }
}

config = {
    'blocks': blocks,
    'geometries': geometries,
    'rules': {
        'sun-avoidance': {
            'min_angle_az': 45,
            'min_angle_alt': 45,
        },     
    },
    'source_targets': [
        ('saturn', 'left', 50),
        ('saturn', 'mid', 50),
        ('saturn', 'right', 50),
    ],
    'merge_order': ['calibration', 'baseline'],
    'az_speed': 1.0, # deg/s
    'az_accel': 2, # deg/s^2
    'elevation': 50, # deg 
    'det_setup_time': 40, # minutes
}

@dataclass(frozen=True)
class SATPolicy(basic.BasePolicy):
    """a more realistic SAT policy. `config` is a string yaml config"""
    blocks: dict
    rules: List[core.Rule]
    geometries: List[dict]
    source_targets: List[tuple]
    
    @classmethod
    def from_config(cls, config: dict):
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
        return core.seq_trim(blocks, t0, t1)

    def transform(self, blocks: core.BlocksTree) -> core.BlocksTree:
        # sun avoidance
        rule = ru.make_rule('sun-avoidance', **self.rules['sun-avoidance'])
        blocks = rule(blocks)
        
        # plan source scans
        cal_blocks = {}
        for source, array_query, el_bore in self.source_targets:
            array_info = inst.array_info_from_query(self.geometries, array_query)
            rule = ru.MakeCESourceScan(
                array_info=array_info, 
                el_bore=el_bore, 
                drift=True
            )
            assert source in blocks['calibration'], f"source {source} not found in sequence"
            if source not in cal_blocks: cal_blocks[source] = []
            cal_blocks[source].append(rule(blocks['calibration'][source]))

        # can we simply merge these blocks for each source?
        for source in cal_blocks:
            if core.seq_has_overlap(cal_blocks[source]):
                # need some special treatment
                # placeholder
                cal_blocks[source] = core.seq_merge(cal_blocks[source], flatten=True)
            else:
                cal_blocks[source] = core.seq_merge(cal_blocks[source], flatten=True)
        
        # store the result back to calibration
        blocks['calibration'] = cal_blocks

        return blocks

    def merge(self, blocks: core.BlocksTree) -> core.Blocks:
        """merge blocks into a single sequence by the order specified
        in self.merge_order, assuming an descending priority order as moving
        down the merge_order list."""
        seq = None
        for query in self.merge_order[::-1]:
            match, _ = core.seq_partition_with_query(query, blocks)
            if seq is None: 
                seq = match
                continue
            else:
                # match takes precedence
                seq = core.seq_merge(seq, match, flatten=True)

        rule = ru.make_rule('min-duration', **self.rules['min-duration'])
        seq = rule(seq)

        return core.seq_sort(seq)

    def seq2cmd(self, seq: core.Blocks):
        """map a scan to a command"""
        commands = [
            "from nextline import disable_trace",
            "",
            "import time",
            "import datetime as dt",
            "",
            "with disable_trace():",
            "    import numpy as np",
            "    import sorunlib as run",
            "    run.initialize()",
            "",
            "UTC = dt.timezone.utc",
            "acu = run.CLIENTS['acu']",
            "pysmurfs = run.CLIENTS['smurf']",
            "",
            "#################################################",
            "",
            f"az_speed = {self.az_speed} # deg/s",
            f"el_speed = {self.el_speed} # deg/s^2",
            f"elevation = {self.elevation} # deg",
            f"det_setup_time = dt.timedelta(minutes={self.det_setup_time})",
            "",
            "#################################################",
            "",
            "# unbias dets",
            "for smurf_instance_id in self.pysmurfs_instance_ids:",
            "    args = ['--bias', '0.0', '--slot', str(smurf_instance_id.split('s')[-1])]",
            "    ok, msg, sess = run.smurf.run.start(script='/readout-script-dev/max/site_control/set_det_biases.py', args=args)",
            "",
            "for smurf in pysmurfs:",
            "    smurf.run.wait()",
            "time.sleep(120)",
            "run.smurf.take_noise(concurrent=True, tag='oper,take_noise,res_check')",
            "",
            "print('Relocking the UFMs')",
            "for smurf in pysmurfs:",
            "    smurf.uxm_relock.start(kwargs={'skip_setup_amps':True})",
            "for smurf in pysmurfs:",
            "    smurf.uxm_relock.wait()",
            "    print(smurf.uxm_relock.status())",
            "",
            "#################################################",
            "",
        ] 
        return cmd.CompositeCommand(commands)