#%%
from dataclasses import dataclass, replace
from typing import List, Union
import datetime as dt

@dataclass(frozen=True)
class Block:
    t0: dt.datetime
    t1: dt.datetime

@dataclass(frozen=True)
class ScanBlock(Block):
    az: float
    alt: float
    throw: float
    patch: str = ""
    comment: str = ""
    
@dataclass(frozen=True)
class Sequence(Block):
    blocks: List[Block]

    @classmethod
    def from_blocks(cls, blocks: List[Block], sort: bool = False) -> 'Sequence':
        if len(blocks) == 0:
            raise ValueError("blocks must be non-empty")
        if sort: blocks = blocks_sort(blocks)
        t0 = min([b.t0 for b in blocks])
        t1 = max([b.t1 for b in blocks])
        return cls(t0=t0, t1=t1, blocks=blocks)

def block_split(block: Block, t: dt.datetime) -> Sequence:
    if t <= block.t0 or t >= block.t1:
        return Sequence.from_blocks([block])
    return Sequence.from_blocks([replace(block, t1=t), replace(block, t0=t)])

def blocks_sort(blocks: List[Block]) -> List[Block]:
    if len(blocks) == 0:
        return []
    elif len(blocks) == 1:
        return blocks
    else:
        return sorted(blocks, key=lambda b: b.t0)
    
def seq_concat(seq1: Sequence, seq2: Sequence) -> Sequence:
    return Sequence.from_blocks(seq1.blocks + seq2.blocks)

def seq_sort(seq: Union[Sequence, List[Block]]) -> Sequence:
    return Sequence.from_blocks(sorted(seq.blocks, key=lambda b: b.t0))

def seq_has_overlap(seq: Sequence) -> bool:
    for i in range(len(seq.blocks)):
        for j in range(i+1, len(seq.blocks)):
            if seq.blocks[i].t1 > seq.blocks[j].t0:
                return True
    return False

def seq_is_sorted(seq: Sequence) -> bool:
    for i in range(len(seq.blocks)-1):
        if seq.blocks[i].t1 > seq.blocks[i+1].t0:
            return False
    return True

def seq_assert_sorted(seq: Sequence) -> None:
    assert seq_is_sorted(seq), "Sequence is not sorted"

def seq_assert_no_overlap(seq: Sequence) -> None:
    assert not seq_has_overlap(seq), "Sequence has overlap"