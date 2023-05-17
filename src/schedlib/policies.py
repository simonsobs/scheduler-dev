#!/usr/bin/env python3

from chex import dataclass
import datetime as dt
from abc import ABC
from typing import List, Dict
from . import core, utils, commands as cmd, instrument as inst, rules as ru, source as src

class RuledBasedPolicy(core.Policy, ABC):
    rules: core.RuleSet
    def make_rule(self, rule_name: str, **kwargs) -> core.Rule:
        assert rule_name in self.rules, f"Rule {rule_name} not found in rules config"
        if not kwargs: kwargs = self.rules[rule_name]  # caller kwargs take precedence
        return ru.make_rule(rule_name, **kwargs)
    def make_multi_rules(self, rule_names: List[str]) -> core.MultiRules:
        return core.MultiRules(rules=[self.make_rule(r) for r in rule_names])

@dataclass(frozen=True)
class BasicPolicy(RuledBasedPolicy):
    master_schedule: str
    calibration_targets: List[str]
    soft_targets: List[str]
    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        master = utils.parse_sequence_from_toast(self.master_schedule)
        calibration = {k: src.source_get_blocks(k, t0, t1) for k in self.calibration_targets}
        soft = {k: src.source_get_blocks(k, t0, t1) for k in self.soft_targets}
        blocks = {
            'master': master,
            'calibration': calibration,
            'soft': soft,
        }
        return core.seq_trim(blocks, t0, t1)

    def apply(self, blocks: core.BlocksTree) -> core.Blocks:
        blocks = self.make_rule("rephase-first")(blocks)
        return core.seq_flatten(blocks)

    def block2cmd(self, block: core.Block):
        if isinstance(block, inst.ScanBlock):
            return cmd.CompositeCommand([
                    f"# {block.name}: {block.patch}",
                    cmd.Goto(block.az, block.alt),
                    cmd.BiasDets(),
                    cmd.Wait(block.t0),
                    cmd.BiasStep(),
                    cmd.Scan(block.patch, block.t1, block.throw),
                    cmd.BiasStep(),
                    "",
            ])
        elif isinstance(block, inst.IVBlock):
            return cmd.IV() 

    def seq2cmd(self, seq: core.Blocks):
        """map a scan to a command"""
        commands = core.seq_flatten(core.seq_map(self.block2cmd, seq))
        commands = [cmd.Preamble()] + commands
        return cmd.CompositeCommand(commands)
