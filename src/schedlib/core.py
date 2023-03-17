from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass(frozen=True)
class Block:
    t0: datetime
    t1: datetime
    alt: float
    az: float
    throw: float
    patch: str

@dataclass(frozen=True)
class ScanBlock(Block):
    pass

@dataclass(frozen=True)
class Sequence:
    blocks: List[Block]
