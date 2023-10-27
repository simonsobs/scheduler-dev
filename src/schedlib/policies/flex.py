import yaml
import os.path as op
from dataclasses import dataclass
import datetime as dt
from typing import List
import jax.tree_util as tu

from . import basic
from .. import config as cfg, core, utils, source as src, rules as ru, commands as cmd, instrument as inst


@dataclass(frozen=True)
class FlexPolicy(basic.BasePolicy):
    """a flexible policy. `config` is a string yaml config *content*"""
    blocks: dict
    rules: List[core.Rule]
    post_rules: List[core.Rule]
    merge_order: List[str]
    geometries: List[dict]

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

        # pre-load the config to populate some common fields in the policy
        # load rules
        rules = []
        for rule_cfg in config.pop('rules', []):
            rule_name = rule_cfg.pop('name')
            # rules that require randomization
            if rule_name in ['make-source-scan', 'rephase-first']:
                today = dt.datetime.now()
                rng_key = utils.PRNGKey((today.year, today.month, today.day, rule_cfg.pop('seed', 0)))
                rule_cfg['rng_key'] = rng_key
                rule = ru.make_rule(rule_name, **rule_cfg)
            # treat special rule
            elif rule_name == 'make-drift-scan':
                rule_cfg['geometries'] = config['geometries']
                block_query = rule_cfg.pop('block_query')
                rule = ru.MakeCESourceScan.from_config(rule_cfg)
                if block_query is not None:
                    rule = ru.ConstrainedRule(rule, block_query)
            else:
                rule = ru.make_rule(rule_name, **rule_cfg)
            rules += [rule]

        # post merge rules tend to be simple
        post_rules = []
        for rule_cfg in config.pop('post_rules', []):
            post_rules.append(ru.make_rule(**rule_cfg))

        # now we can construct the policy
        return cls(rules=rules, post_rules=post_rules, **config)

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
        # apply each rule
        for rule in self.rules: 
            blocks = rule(blocks)
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

        # apply transformation if needed
        for rule in self.post_rules:
            seq = rule(seq)

        return core.seq_sort(seq)

    def block2cmd(self, block: core.Block):
        if isinstance(block, inst.ScanBlock):
            return cmd.CompositeCommand([
                    f"# {block.name}",
                    cmd.Goto(block.az, block.alt),
                    cmd.BiasDets(),
                    cmd.Wait(block.t0),
                    cmd.BiasStep(),
                    cmd.Scan(block.name, block.t1, block.throw, block.az_drift),
                    cmd.BiasStep(),
                    "",
            ])

    def seq2cmd(self, seq: core.Blocks):
        """map a scan to a command"""
        commands = core.seq_flatten(core.seq_map(self.block2cmd, seq))
        commands = [cmd.Preamble()] + commands
        return cmd.CompositeCommand(commands)

    def get_drift_scans(self, t0, t1, el_bore, array_query):
        """a convenience function to build drift source scans from a policy.
        
        Parameters
        ----------
        t0 : datetime
            start time
        t1 : datetime
            end time
        el_bore : float
            elevation of the boresight in degrees 
        array_query : str
            query for the part of array to focus

        """
        # construct the sequence
        seqs = self.init_seqs(t0, t1)
        # find the subset of array of interests based on the query
        array_info = inst.array_info_from_query(self.geometries, array_query)
        # construct a rule that does the transformation
        rule = ru.MakeCESourceScan(array_info=array_info, el_bore=el_bore, drift=True)
        # apply the rule
        return rule(seqs)