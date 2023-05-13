#%%
from typing import List, Union, Callable, Optional, Any
import datetime as dt
from chex import dataclass

@dataclass(frozen=True)
class Block:
    t0: dt.datetime
    t1: dt.datetime

    @property
    def duration(self) -> dt.timedelta:
        return self.t1 - self.t0

Blocks = List[Block]
MaybeBlock = Union[Block, None]
MaybeBlocks = List[MaybeBlock]

@dataclass(frozen=True)
class ScanBlock(Block):
    az: float
    alt: float
    throw: float
    patch: str
    
def block_split(block: Block, t: dt.datetime) -> Blocks:
    if t <= block.t0 or t >= block.t1:
        return [block]
    return [block.replace(t1=t), block.replace(t0=t)]

def block_trim(block: Block, t0: Optional[dt.datetime] = None, t1: Optional[dt.datetime] = None) -> MaybeBlock:
    t0 = t0 or block.t0
    t1 = t1 or block.t1
    if t0 >= block.t1 or t1 <= block.t0:
        return None
    return block.replace(t0=max(block.t0, t0), t1=min(block.t1, t1))

def block_shift(block: Block, dt: dt.timedelta) -> Block:
    return block.replace(t0=block.t0+dt, t1=block.t1+dt)

def block_extend(block: Block, dt: dt.timedelta) -> Block:
    return block.replace(t0=block.t0-dt/2, t1=block.t1+dt/2)  # note dt/2

def block_extend_left(block: Block, dt: dt.timedelta) -> Block:
    return block.replace(t0=block.t0-dt, t1=block.t1)

def block_extend_right(block: Block, dt: dt.timedelta) -> Block:
    return block.replace(t0=block.t0, t1=block.t1+dt)

def block_shrink(block: Block, dt: dt.timedelta) -> MaybeBlock:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0+dt/2, t1=block.t1-dt/2)  # note dt/2

def block_shrink_left(block: Block, dt: dt.timedelta) -> MaybeBlock:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0+dt, t1=block.t1)

def block_shrink_right(block: Block, dt: dt.timedelta) -> MaybeBlock:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0, t1=block.t1-dt)

def block_trim_left_to(block: Block, t: dt.datetime) -> MaybeBlock:
    if t >= block.t1:
        return None
    return block.replace(t0=max(block.t0, t))
    
def seq_sort(seq: Blocks) -> Blocks:
    return sorted(seq, key=lambda b: b.t0)

def seq_drop_empty(blocks: MaybeBlocks) -> Blocks:
    return list(filter(lambda block: block is not None, blocks))

def seq_has_overlap(blocks: Blocks) -> bool:
    blocks = seq_sort(blocks)
    for i in range(len(blocks)-1):
        if blocks[i].t1 > blocks[i+1].t0:
            return True
    return False

def seq_is_sorted(blocks: Blocks) -> bool:
    for i in range(len(blocks)-1):
        if blocks[i].t1 > blocks[i+1].t0:
            return False
    return True

def seq_assert_sorted(blocks: Blocks) -> None:
    assert seq_is_sorted(blocks), "Sequence is not sorted"

def seq_assert_no_overlap(seq: Blocks) -> None:
    assert not seq_has_overlap(seq), "Sequence has overlap"

def seq_filter(op: Callable[[MaybeBlock], bool], blocks: MaybeBlocks) -> MaybeBlocks:
    return list(filter(op, blocks))

def seq_map(op: Callable[[MaybeBlock], Any], blocks: MaybeBlocks) -> List[Any]:
    return list(map(op, blocks))

def seq_map_when(op_when: Callable[[MaybeBlock], bool], op: Callable[[Block], Any], blocks: MaybeBlocks) -> Any:
    return list(map(lambda block: op(block) if op_when(block) else block, blocks))
