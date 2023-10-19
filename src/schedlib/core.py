from typing import List, Union, Callable, Optional, Any, TypeVar
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import jax.tree_util as tu
import equinox
from dataclasses import dataclass, replace as dc_replace

from . import utils

@dataclass(frozen=True)
class Block:
    t0: dt.datetime
    t1: dt.datetime
    @property
    def duration(self) -> dt.timedelta:
        return self.t1 - self.t0
    def split(self, t: dt.datetime) -> List["Block"]:
        return block_split(self, t)
    def trim(self, t0: Optional[dt.datetime] = None, t1: Optional[dt.datetime] = None) -> List["Block"]:
        return block_trim(self, t0, t1)
    def shift(self, dt: dt.timedelta) -> "Block":
        return block_shift(self, dt)
    def extend(self, dt: dt.timedelta) -> "Block":
        return block_extend(self, dt)
    def extend_left(self, dt: dt.timedelta) -> "Block":
        return block_extend_left(self, dt)
    def extend_right(self, dt: dt.timedelta) -> "Block":
        return block_extend_right(self, dt)
    def shrink(self, dt: dt.timedelta) -> List["Block"]:
        return block_shrink(self, dt)
    def shrink_left(self, dt: dt.timedelta) -> List["Block"]:
        return block_shrink_left(self, dt)
    def shrink_right(self, dt: dt.timedelta) -> List["Block"]:
        return block_shrink_right(self, dt)
    def trim_left_to(self, t: dt.datetime) -> List["Block"]:
        return block_trim_left_to(self, t)
    def trim_right_to(self, t: dt.datetime) -> List["Block"]:
        return block_trim_right_to(self, t)
    def isa(self, block_type: "BlockType") -> bool:
        return block_isa(block_type)(self)
    def replace(self, **kwargs) -> "Block":
        return dc_replace(self, **kwargs)

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

def block_trim_right_to(block: Block, t: dt.datetime) -> Blocks:
    if t <= block.t0:
        return None
    return block.replace(t1=min(block.t1, t))

def block_isa(block_type:BlockType) -> Callable[[Block], bool]:
    def isa(block: Block) -> bool:
        return isinstance(block, block_type)
    return isa

def block_overlap(block1: Block, block2: Block) -> bool:
    return (block1.t0 - block2.t1).total_seconds() * (block1.t1 - block2.t0).total_seconds() < 0

def block_merge(block1: Block, block2: Block) -> Blocks:
    """merge block2 into block1. It will return a sorted seq"""
    if not block_overlap(block1, block2):
        return seq_sort([block1, block2])
    else: # handle merge
        if block2.t0 < block1.t0:
            return seq_sort([block2, block1.trim_left_to(block2.t1)])
        else:
            return seq_sort([block1.trim_right_to(block2.t0), block2, block1.trim_left_to(block2.t1)])

# =============================
# Sequence / Blocks operations 
# =============================

def seq_is_nested(blocks: Blocks) -> bool:
    is_leaf = lambda x: is_block(x) or x is None
    return not tu.all_leaves(blocks, is_leaf=is_leaf)

def seq_assert_not_nested(blocks: Blocks) -> None:
    assert not seq_is_nested(blocks), "seq has nested blocks"

def seq_sort(seq: Blocks, flatten=False, key_fn=lambda b: b.t0) -> Blocks:
    """sort is only implemented for flat seq. This function will flatten the seq
    if flatten=True, it will raise error if seq is nested. The flatten option 
    is to enforce that we should be explicit when nestedness is destroyed."""
    if seq_is_nested(seq) and not flatten:
        raise ValueError("Cannot sort nested sequence, use flatten=True")
    return sorted(seq_flatten(seq), key=key_fn)

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

def seq_has_overlap_with_block(seq: Blocks, block: Block) -> bool:
    for b in seq_flatten(seq):
        if block_overlap(b, block):
            return True
    return False

def seq_merge_block(seq: Blocks, block: Block, flatten=False) -> Blocks:
    if not flatten and seq_is_nested(seq):
        raise ValueError("Cannot merge block into nested sequence, use flatten=True")
    if not seq: return [block]
    if not seq_has_overlap_with_block(seq, block): return seq_sort(seq + [block])
    return seq_drop_duplicates(
        seq_map_when(
            lambda b: block_overlap(b, block),
            lambda b: block_merge(b, block), 
            seq), 
        flatten=True)

def seq_drop_duplicates(seq: Blocks, flatten=False, sort=True) -> Blocks:
    if not flatten and seq_is_nested(seq):
        raise ValueError("Cannot drop duplicates in nested sequence, use flatten=True")
    res = list(set(seq_flatten(seq)))
    if sort: return seq_sort(res)
    return res
    
def seq_merge(seq1: Blocks, seq2: Blocks, flatten=False) -> Blocks:
    """merge seq2 into seq1. Both seqs will be flattened before merging,
    and in the case of conflict, the seq2 will take precedence, and
    seq1 blocks will be overwritten or splitted, whenever necessary."""
    if not flatten and (seq_is_nested(seq1) or seq_is_nested(seq2)):
        raise ValueError("Cannot merge nested sequence, use flatten=True")
    # fixme: it's possible that seq1 and seq2 have overlap, need to consider that
    seq1 = seq_flatten(seq1)
    seq2 = seq_flatten(seq2)
    seq = seq1
    # lazy implementation for now
    for block in seq2:
        seq = seq_merge_block(seq, block)
    return seq_sort(seq)

# =========================
# Tree related
# =========================

# placeholder type var for readability: a nested tree (dict, tuple, list) of blocks
BlocksTree = TypeVar('BlocksTree')

def seq_treedef(blocks: BlocksTree, include_none=False) -> tu.PyTreeDef:
    if not include_none:
        return tu.tree_structure(blocks, is_leaf=is_block)
    else:
        return tu.tree_structure(blocks, is_leaf=lambda x: is_block(x) or x is None)

def seq_flatten(blocks: BlocksTree) -> Blocks:
    """Flatten nested blocks into a single list of books and drop Nones"""
    return tu.tree_leaves(blocks, is_leaf=is_block)

def seq_unflatten(treedef: tu.PyTreeDef, blocks: Blocks) -> BlocksTree:
    return tu.tree_unflatten(treedef, blocks)

def seq_assert_same_structure(*trees: BlocksTree) -> None:
    treedefs = [seq_treedef(t, include_none=True) for t in trees]
    assert all(t1 == t2 for t1, t2 in zip(treedefs, treedefs[1:])), "Trees have different structure"

def seq_filter(op: Callable[[Block], bool], blocks: BlocksTree) -> BlocksTree:
    return tu.tree_map(lambda b: None if not op(b) else b, blocks, is_leaf=is_block)

def seq_filter_out(op: Callable[[Block], bool], blocks: BlocksTree) -> BlocksTree:
    return tu.tree_map(lambda b: None if op(b) else b, blocks, is_leaf=is_block)

def seq_map(op, *blocks: BlocksTree) -> List[Any]:
    return tu.tree_map(op, *blocks, is_leaf=is_block)

def seq_map_with_path(op, *blocks: BlocksTree) -> List[Any]:
    return tu.tree_map_with_path(op, *blocks, is_leaf=is_block)

def seq_map_when(op_when: Callable[[Block], bool], op: Callable[[Block], Any], blocks: BlocksTree) -> List[Any]:
    return tu.tree_map(lambda b: op(b) if op_when(b) else b, blocks, is_leaf=is_block)

def seq_replace_block(blocks: BlocksTree, source: Block, target: Block) -> BlocksTree:
    return seq_map_when(lambda b: b == source, lambda _: target, blocks)

def seq_trim(blocks: BlocksTree, t0: dt.datetime, t1: dt.datetime) -> BlocksTree:
    return seq_map(lambda b: b.trim(t0, t1), blocks)

def seq_partition(op, blocks: BlocksTree) -> List[Any]:
    """partition a blockstree into two trees, one for blocks that satisfy the predicate,
    which is specified through a function that takes a block as input and returns
    a boolean, and the second return is for blocks that don't match the predicate. Unmatched
    values will be left as None."""
    filter_spec = tu.tree_map(op, blocks, is_leaf=is_block)
    return equinox.partition(blocks, filter_spec)

def seq_partition_with_path(op, blocks: BlocksTree, **kwargs) -> List[Any]:
    """partition a blockstree into two trees, one for blocks that satisfy the predicate,
    which is specified through a function that takes a block and path as input and returns
    a boolean, and the second return is for blocks that don't match the predicate. Unmatched
    values will be left as None."""
    filter_spec = tu.tree_map_with_path(op, blocks, is_leaf=is_block)
    return equinox.partition(blocks, filter_spec, **kwargs)

def seq_partition_with_query(query, blocks: BlocksTree):
    def path2key(path):
        """convert a path (used in tree_util.tree_map_with_path) to a dot-separated key"""
        keys = []
        for p in path:
            if isinstance(p, tu.SequenceKey):
                keys.append(p.idx)
            elif isinstance(p, tu.DictKey):
                keys.append(p.key)
            else:
                raise ValueError(f"unknown path type {type(p)}")
        return ".".join([str(k) for k in keys])
    return seq_partition_with_path(lambda path, block: utils.match_query(path, query), blocks)

def seq_combine(*blocks: BlocksTree) -> BlocksTree:
    """combine blocks from multiple trees into a single tree, where the blocks are
    combined in a list. The trees must have the same structure."""
    seq_assert_same_structure(*blocks)
    return equinox.combine(*blocks, is_leaf=is_block)

# =========================
# Other useful Block types
# =========================

@dataclass(frozen=True)
class NamedBlock(Block):
    name: str 

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
class Policy(BlocksTransformation, ABC):
    """apply: apply policy to a tree of blocks"""
    # initialize a tree of blocks
    @abstractmethod
    def init_seqs(self) -> BlocksTree: ...
    @abstractmethod
    def apply(self, blocks: BlocksTree) -> Blocks: ...

# ===============================
# Others convenience types alias
# ===============================

Arr = np.ndarray
