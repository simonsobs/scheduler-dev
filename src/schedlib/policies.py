#!/usr/bin/env python3

from chex import dataclass
import datetime as dt
from abc import ABC
from typing import List
from . import core, utils, commands as cmd, instrument as inst, rules as ru, source as src

@dataclass(frozen=True)
class BasePolicy(core.Policy, ABC):
    rules: core.RuleSet
    def make_rule(self, rule_name: str, **kwargs) -> core.Rule:
        if not kwargs:
            assert rule_name in self.rules, f"Rule {rule_name} not found in rules config"
            kwargs = self.rules[rule_name]  # caller kwargs take precedence
        return ru.make_rule(rule_name, **kwargs)
    def make_multi_rules(self, rule_names: List[str]) -> core.MultiRules:
        return core.MultiRules(rules=[self.make_rule(r) for r in rule_names])

@dataclass(frozen=True)
class BasicPolicy(BasePolicy):
    master_schedule: str
    calibration_targets: List[str]
    soft_targets: List[str]

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        master = utils.parse_sequence_from_toast(self.master_schedule)
        calibration = {k: src.source_gen_seq(k, t0, t1) for k in self.calibration_targets}
        soft = {k: src.source_gen_seq(k, t0, t1) for k in self.soft_targets}
        blocks = {
            'master': master,
            'sources': {
                'calibration': calibration,
                'soft': soft,
            }
        }
        return core.seq_trim(blocks, t0, t1)

    def apply(self, blocks: core.BlocksTree) -> core.Blocks:
        # sun avoidance for all
        blocks = self.make_rule('sun-avoidance')(blocks)

        # plan for sources
        blocks['sources'] = self.make_rule('make-source-plan')(blocks['sources'])

        # add calibration targets
        cal_blocks = blocks['sources']['calibration']
        if 'day-mod' in self.rules:
            cal_blocks = self.make_rule('day-mod')(cal_blocks)
        if 'drift-mode' in self.rules:
            cal_blocks = self.make_rule('drift-mode')(cal_blocks)
        if 'min-duration-cal' in self.rules:
            cal_blocks = self.make_rule(
                'min-duration',
                **self.rules['min-duration-cal']
            )(cal_blocks)

        # actually turn observation windows into source scans: need some random
        # numbers to rephase each source scan in an observing window. we will
        # use a daily static key, producing exactly the same sequence of random
        # numbers when the date is the same
        if len(core.seq_sort(cal_blocks, flatten=True)) > 0:
            first_block = core.seq_sort(cal_blocks, flatten=True)[0]
            keys = utils.daily_static_key(first_block.t0).split(len(cal_blocks))
            for srcname, key in zip(cal_blocks, keys):
                cal_blocks[srcname] = self.make_rule(
                    'make-source-scan',
                    rng_key=key,
                    **self.rules['make-soure-scan']
                )(cal_blocks[srcname])

        # merge all sources into main sequence
        blocks = core.seq_merge(blocks['master'], cal_blocks, flatten=True)
        return core.seq_sort(blocks)

    def block2cmd(self, block: core.Block):
        if isinstance(block, inst.ScanBlock):
            return cmd.CompositeCommand([
                    f"# {block.name}",
                    cmd.Goto(block.az, block.alt),
                    cmd.BiasDets(),
                    cmd.Wait(block.t0),
                    cmd.BiasStep(),
                    cmd.Scan(block.name, block.t1, block.throw),
                    cmd.BiasStep(),
                    "",
            ])

    def seq2cmd(self, seq: core.Blocks):
        """map a scan to a command"""
        commands = core.seq_flatten(core.seq_map(self.block2cmd, seq))
        commands = [cmd.Preamble()] + commands
        return cmd.CompositeCommand(commands)
