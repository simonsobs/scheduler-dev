from typing import List
from dataclasses import dataclass, field
from datetime import datetime

from .core import ScanBlock, Blocks

@dataclass(frozen=True)
class Command:
    pass

@dataclass(frozen=True)
class Goto(Command):
    az: float
    alt: float
    def __str__(self):
        return f"acu.move_to({self.az:.2f}, {self.alt:.2f})"

@dataclass(frozen=True)
class Scan(Command):
    field: str    
    stop: datetime
    width: float
    def __str__(self):
        return f"seq.scan(description='{self.field}', stop_time='{self.stop.isoformat()}', width={self.width:.2f})"

@dataclass(frozen=True)
class Wait(Command):
    t0: datetime
    def __str__(self):
        return f"wait_until('{self.t0.isoformat()}')"

@dataclass(frozen=True)
class BiasStep(Command):
    def __str__(self):
        return "smurf.bias_step()"
    
@dataclass(frozen=True)
class Stream(Command):
    state: str
    def __str__(self):
        return f"smurf.stream('{self.state}')"

@dataclass(frozen=True)
class CompositeCommand(Command):
    commands: List[Command]
    def __str__(self):
        return "\n".join([str(cmd) for cmd in self.commands])

@dataclass(frozen=True)
class IVCurve(Command):
    def __str__(self):
        return "smurf.bias_step()"

@dataclass(frozen=True)
class BiasDets(Command):
    def __str__(self):
        return "smurf.bias_dets()"        
    
@dataclass(frozen=True)
class IV(CompositeCommand):
    commands: List[Command] = field(default_factory=[
        IVCurve(),
        BiasDets()
    ])

@dataclass(frozen=True)
class Preamble(CompositeCommand):
    commands: List[Command] = field(default_factory=lambda: [
        "from sorunlib import *",
        "from nextline import disable_trace",
        "",
        "with disable_trace():",
        "\tinitialize(test_mode=True)",
        "",
        "smurf.uxm_setup()",
        "smurf.iv_curve()",
        ""
    ])

def seq2cmd(seq: Blocks):
    """map a scan to a command"""
    commands = [Preamble()]
    for block in seq:
        if block is None: 
            raise ValueError("None block in sequence")
        if isinstance(block, ScanBlock):
            command = CompositeCommand([
                f"# {block.patch}",
                Goto(block.az, block.alt),
                BiasDets(),
                Wait(block.t0),
                BiasStep(),
                # Stream('on'),
                Scan(block.patch, block.t1, block.throw),
                # Stream('off'),
                BiasStep(),
                "",  # line break
            ])
        else: 
            raise ValueError(f"Unknown block type {type(block)}")
        commands.append(command)
    return CompositeCommand(commands)
