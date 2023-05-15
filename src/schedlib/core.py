from typing import List, Union, Callable, Optional, Any
from chex import dataclass
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
from toolz import compose_left
import jax.tree_util as tu

@dataclass(frozen=True)
class Block:
    t0: dt.datetime
    t1: dt.datetime
    @property
    def duration(self) -> dt.timedelta:
        return self.t1 - self.t0

BlockType = type(Block)
Blocks = List[Union[Block, None, "Blocks"]]  # maybe None, maybe nested

def is_block(x: Any) -> bool:
    return isinstance(x, Block)

def block_split(block: Block, t: dt.datetime) -> Blocks:
    if t <= block.t0 or t >= block.t1:
        return [block]
    return [block.replace(t1=t), block.replace(t0=t)]

def block_trim(block: Block, t0: Optional[dt.datetime] = None, t1: Optional[dt.datetime] = None) -> Blocks:
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

def block_shrink(block: Block, dt: dt.timedelta) -> Blocks:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0+dt/2, t1=block.t1-dt/2)  # note dt/2

def block_shrink_left(block: Block, dt: dt.timedelta) -> Blocks:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0+dt, t1=block.t1)

def block_shrink_right(block: Block, dt: dt.timedelta) -> Blocks:
    if block.duration <= dt:
        return None
    return block.replace(t0=block.t0, t1=block.t1-dt)

def block_trim_left_to(block: Block, t: dt.datetime) -> Blocks:
    if t >= block.t1:
        return None
    return block.replace(t0=max(block.t0, t))

def block_isa(block_type:BlockType) -> Callable[[Block], bool]:
    def isa(block: Block) -> bool:
        return isinstance(block, block_type)
    return isa

# =============================
# Sequence / Blocks operations 
# =============================

def seq_is_nested(blocks: Blocks) -> bool:
    for block in blocks:
        if isinstance(block, list):
            return True
    return False

def seq_assert_not_nested(blocks: Blocks) -> None:
    assert not seq_is_nested(blocks), "seq has nested blocks"

def seq_flatten(blocks: Blocks) -> Blocks:
    """Flatten nested blocks into a single list of books and drop Nones"""
    return tu.tree_leaves(blocks, is_leaf=is_block)

def seq_sort(seq: Blocks, flatten=False) -> Blocks:
    if seq_is_nested(seq) and not flatten:
        raise ValueError("Cannot sort nested sequence, use flatten=True")
    return sorted(seq_flatten(seq), key=lambda b: b.t0)

def seq_has_overlap(blocks: Blocks) -> bool:
    blocks = seq_sort(blocks, flatten=True)
    for i in range(len(blocks)-1):
        if blocks[i].t1 > blocks[i+1].t0:
            return True
    return False

def seq_is_sorted(blocks: Blocks) -> bool:
    blocks = seq_flatten(blocks)
    for i in range(len(blocks)-1):
        # only care about causal ordering
        if blocks[i].t0 > blocks[i+1].t0:
            return False
    return True

def seq_assert_sorted(blocks: Blocks) -> None:
    assert seq_is_sorted(blocks), "Sequence is not sorted"

def seq_assert_no_overlap(seq: Blocks) -> None:
    assert not seq_has_overlap(seq), "Sequence has overlap"

def seq_filter(op: Callable[[Blocks], bool], blocks: Blocks) -> Blocks:
    return tu.tree_map(lambda b: None if not op(b) else b, blocks, is_leaf=is_block)

def seq_filter_out(op: Callable[[Blocks], bool], blocks: Blocks) -> Blocks:
    return tu.tree_map(lambda b: None if op(b) else b, blocks, is_leaf=is_block)

def seq_map(op: Callable[[Blocks], Any], blocks: Blocks) -> List[Any]:
    """preserves nesting and nones"""
    return tu.tree_map(op, blocks, is_leaf=is_block)

def seq_map_when(op_when: Callable[[Blocks], bool], op: Callable[[Block], Any], blocks: Blocks) -> List[Any]:
    return tu.tree_map(lambda b: op(b) if op_when(b) else b, blocks, is_leaf=is_block)

def seq_trim(blocks: Blocks, t0: dt.datetime, t1: dt.datetime) -> Blocks:
    return seq_map(lambda b: block_trim(b, t0, t1), blocks)

# =========================
# Other useful Block types
# =========================

@dataclass(frozen=True)
class ScanBlock(Block):
    az: float
    alt: float
    throw: float
    patch: str

# =========================
# Rules and Policies
# =========================

@dataclass(frozen=True)
class BlocksTransformation(ABC):
    @abstractmethod
    def apply(self, blocks: Blocks) -> Blocks: ...
    def __call__(self, blocks: Blocks) -> Blocks:
        """wrapper to make it compatible with callable functions"""
        return self.apply(blocks)

Rule = Union[BlocksTransformation, Callable[[Blocks], Blocks]]
RuleSet = List[Rule]

@dataclass(frozen=True)
class Policy:
    rules: RuleSet
    def apply(self, blocks: Blocks) -> Blocks:
        """apply rules to blocks in first-to-last order"""
        return compose_left(*self.rules)(blocks)

# ===============================
# Others convenience types alias
# ===============================

Arr = np.ndarray