#!/usr/bin/env python3
from chex import dataclass
from . import core

@dataclass(frozen=True)
class ScanBlock(core.Block):
    az: float
    alt: float
    throw: float
    patch: str

class FixedBlock(core.Block): pass
