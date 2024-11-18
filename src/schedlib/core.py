from typing import List, Union, Callable, Optional, Any, TypeVar
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import jax.tree_util as tu
from dataclasses import dataclass, replace as dc_replace, asdict
from functools import reduce

from . import utils

@dataclass(frozen=True)
class Block:
    """Block is a time interval with a start time and an end time. 
    It's immutable."""
    t0: dt.datetime
    t1: dt.datetime
    @property
    def duration(self) -> dt.timedelta:
        return self.t1 - self.t0
    def split(self, t: dt.datetime) -> List["Block"]:
        return block_split(self, t)
    def split_n(self, dt: dt.timedelta) -> List["Block"]:
        return block_split_n(self, dt)
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
    def isa(self, block_type) -> bool:
        return block_isa(block_type)(self)
    def replace(self, **kwargs) -> "Block":
        return dc_replace(self, **kwargs)
    def overlaps(self, block: "Block") -> bool:
        return block_overlap(self, block)
    def to_dict(self):
        # unlike __dict__, it makes deep copy
        return asdict(self)

BlockType = type(Block)
Blocks = List[Union[Block, None, "Blocks"]]  # maybe None, maybe nested

@dataclass(frozen=True)
class NamedBlock(Block):
    name: str

def is_block(x: Any) -> bool:
    return isinstance(x, Block)

def block_split(block: Block, t: dt.datetime) -> Blocks:
    """split a block into two blocks at time t. If t is outside the block,
    the original block will be returned.

    Parameters
    ----------
    block : Block
        the block to be split
    t : dt.datetime
        the time to split

    Returns
    -------
    Blocks
        a list of blocks, either the original block or two blocks split at time t
    """
    if t <= block.t0 or t >= block.t1:
        return [block]
    return [block.replace(t1=t), block.replace(t0=t)]

def block_split_n(block: Block, dt: dt.timedelta) -> Blocks:
    """split a block into a series of smaller blocks of length
    dt

    Parameters
    ----------
    block : Block
        the block to be split
    dt : dt.timedelta
        the duration of the blocks to split into

    Returns
    -------
    Blocks
        a list of blocks, either the original block or multiple blocks of length dt
    """
    t0_off = block.t0 + dt
    
    blocks_left = block_split(block, t0_off)

    blocks = []
    blocks.append(blocks_left[0])

    while len(blocks_left) > 1:
        t0_off += dt
        blocks_left = block_split(blocks_left[-1], t0_off)
        blocks.append(blocks_left[0])
    return blocks

def block_trim(block: Block, t0: Optional[dt.datetime] = None, t1: Optional[dt.datetime] = None) -> Blocks:
    """trim a block to the given time range. If the time range is outside the block,
    None will be returned.
    
    Parameters
    ----------
    block : Block
        the block to be trimmed
    t0 : Optional[dt.datetime], optional
        the start time of the trimmed block, by default None
    t1 : Optional[dt.datetime], optional
        the end time of the trimmed block, by default None
    
    Returns
    -------
    Blocks
        a list of blocks, either the original block or the trimmed block
    """
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

def block_isa(block_type) -> Callable[[Block], bool]:
    def isa(block: Block) -> bool:
        return isinstance(block, block_type)
    return isa

def block_overlap(block1: Block, block2: Block) -> bool:
    """check if two blocks overlap
    
    Parameters
    ----------
    block1 : Block
        the first block
    block2 : Block
        the second block
    
    Returns
    -------
    bool
        True if the two blocks overlap, False otherwise
    """
    return (block1.t0 - block2.t1).total_seconds() * (block1.t1 - block2.t0).total_seconds() < 0

def block_intersect(block1: Block, block2: Block) -> Block:
    """get the intersection of two blocks

    Parameters
    ----------
    block1 : Block
        the first block
    block2 : Block
        the second block

    Returns
    -------
    Block
        the intersection of the two blocks
    """
    return block_trim(block1, t0=block2.t0, t1=block2.t1)

def block_merge(block1: Block, block2: Block) -> Blocks:
    """merge block2 into block1. It will return a sorted seq. block2 will take
    precedence if there is overlap, and block1 will be overwritten or splitted
    whenever necessary.
    
    Parameters
    ----------
    block1 : Block
        the first block
    block2 : Block
        the second block
        
    Returns
    -------
    Blocks
        a sorted seq of blocks
    """
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
    """check if a sequence is nested, i.e. contains blocks that are not leaves
    or None
    
    Parameters
    ----------
    blocks : Blocks
        a sequence of blocks
    
    Returns
    -------
    bool
        True if the sequence is nested, False otherwise
    """
    # special cases: if it's a block or None, it's not nested
    if is_block(blocks): return False
    if blocks is None: return False
    is_leaf = lambda x: is_block(x) or x is None
    return not tu.all_leaves(blocks, is_leaf=is_leaf)

def seq_assert_not_nested(blocks: Blocks) -> None:
    assert not seq_is_nested(blocks), "seq has nested blocks"

def seq_sort(seq: Blocks, flatten=False, key_fn=lambda b: b.t0) -> Blocks:
    """sort a list of blocks. Only implemented for flat seq. This function 
    will flatten the seq if flatten=True, it will raise error if seq is 
    nested. The flatten option is to enforce that we should be explicit 
    when nestedness is destroyed.
    
    Parameters
    ----------
    seq : Blocks
        a sequence of blocks to be sorted
    flatten : bool, optional
        whether to flatten the seq before sorting, by default False
    key_fn : Callable[[Block], Any], optional
        the key function to sort the blocks, by default lambda b: b.t0

    Returns
    -------
    Blocks
        a sorted sequence of blocks
    """
    if seq_is_nested(seq) and not flatten:
        raise ValueError("Cannot sort nested sequence, use flatten=True")
    seq = seq_flatten(seq)

    # preserve causal ordering
    order = np.arange(len(seq))
    seq = [x[0] for x in sorted(zip(seq, order), key=lambda x: (key_fn(x[0]), x[1]))]
    return seq

def seq_has_overlap(blocks: Blocks) -> bool:
    """check if a sequence has overlap between blocks
    
    Parameters
    ----------
    blocks : Blocks
        a sequence of blocks
        
    Returns
    -------
    bool
        True if the sequence has overlap, False otherwise
    """
    blocks = seq_sort(blocks, flatten=True)
    for i in range(len(blocks)-1):
        if blocks[i].t1 > blocks[i+1].t0:
            return True
    return False

def seq_is_sorted(blocks: Blocks, key_fn=lambda b: b.t0) -> bool:
    """check if a sequence is sorted according to the key function
    
    Parameters
    ----------
    blocks : Blocks
        a sequence of blocks
    key_fn : Callable[[Block], Any], optional
        the key function to sort the blocks, by default lambda b: b.t0
    
    Returns
    -------
    bool
        True if the sequence is sorted, False otherwise
    """
    blocks = seq_flatten(blocks)
    for i in range(len(blocks)-1):
        # only care about causal ordering
        if key_fn(blocks[i]) > key_fn(blocks[i+1]):
            return False
    return True

def seq_assert_sorted(blocks: Blocks) -> None:
    assert seq_is_sorted(blocks), "Sequence is not sorted"

def seq_assert_no_overlap(seq: Blocks) -> None:
    assert not seq_has_overlap(seq), "Sequence has overlap"

def seq_has_overlap_with_block(seq: Blocks, block: Block, allowance: int = 0) -> bool:
    """check if a sequence has overlap with a block
    
    Parameters
    ----------
    seq : Blocks
        a sequence of blocks
    block : Block
        a block to check overlap with
    allowance: int
        minimum overlap to be considered overlap in seconds
        
    Returns
    -------
    bool
        True if the sequence has overlap with the block, False otherwise
    """
    # if we pass in a non-zero allowance, it effectively acts as
    # a smaller block on both side.
    if allowance > 0:
        block = block.shrink(dt.timedelta(seconds=2*allowance))  # shrink go by total duration so 2*allowance for left and right.
    for b in seq_flatten(seq):
        if block_overlap(b, block):
            return True
    return False

def seq_merge_block(seq: Blocks, block: Block, flatten=False) -> Blocks:
    """merge a block into a sequence of blocks. If the block has overlap with
    any block in the sequence, the overlapping blocks will be merged, and the
    new block will take precedence. The sequence will be sorted after merging.
    As usual, flatten is to acknowledge that we are destroying nestedness.
    
    Parameters
    ----------
    seq : Blocks
        a sequence of blocks
    block : Block
        a block to be merged into the sequence
    flatten : bool, optional
        whether to flatten the seq before merging, by default False 

    Returns
    -------
    Blocks
        a sorted sequence of blocks
    """
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

def seq_resolve_overlap(seq: Blocks, reverse=False):
    """merge blocks in sequentially to resolve conflict in a flattened list of blocks.
    Block that comes later in the seq always take higher priority. If reverse is true,
    the merging will happen in reverse order, and the block that comes earlier in the seq
    always take priority in merging"""
    seq_assert_not_nested(seq)
    if reverse: seq = seq[::-1]
    return seq_sort(reduce(lambda carry, b: seq_merge_block(carry, b), seq, []),
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
    seq1 blocks will be overwritten or splitted, whenever necessary.
    
    Parameters
    ----------
    seq1 : Blocks
        a sequence of blocks
    seq2 : Blocks
        a sequence of blocks to be merged into seq1
    flatten : bool, optional
        whether to flatten the seqs before merging, by default False

    Returns
    -------
    Blocks
        a sorted sequence of blocks
    """
    if not flatten and (seq_is_nested(seq1) or seq_is_nested(seq2)):
        raise ValueError("Cannot merge nested sequence, use flatten=True")
    # fixme: it's possible that seq1 and seq2 have overlap, need to consider that
    assert not seq_has_overlap(seq1) and not seq_has_overlap(seq2), "one of the seqs has internal overlap which is not supported yet"
    seq1 = seq_flatten(seq1)
    seq2 = seq_flatten(seq2)
    seq = seq1
    # lazy implementation for now
    for block in seq2:
        seq = seq_merge_block(seq, block)
    return seq_sort(seq)

def seq_remove_overlap(seq1, seq2, flatten=False):
    """return a copy of seq with regions that overlap with seq2 removed"""
    seq2 = seq_map(lambda b: NamedBlock(name="_dummy", t0=b.t0, t1=b.t1), seq2)
    return seq_flatten(seq_filter_out(
        lambda b: isinstance(b, NamedBlock) and b.name=="_dummy",
        seq_merge(seq1, seq2, flatten=flatten)
    ))

# =========================
# Tree related
# =========================

# placeholder type var for readability: a nested tree (dict, tuple, list) of blocks
BlocksTree = TypeVar('BlocksTree')

def seq_treedef(blocks: BlocksTree, include_none=False) -> tu.PyTreeDef:
    """get the tree structure of a tree of blocks, which is a tuple of the same
    structure as the tree, but with the blocks replaced by None. This is useful
    for checking if two trees have the same structure, which is a prerequisite
    for some tree operations such as tree_combine. If include_none=True, then
    the None values will be included in the tree structure, otherwise they will
    be dropped.
    
    Parameters
    ----------
    blocks : BlocksTree
        a tree of blocks
    include_none : bool, optional
        whether to include None in the tree structure, by default False

    Returns
    -------
    tu.PyTreeDef
        the tree structure of the tree of blocks
    """
    if not include_none:
        return tu.tree_structure(blocks, is_leaf=is_block)
    else:
        return tu.tree_structure(blocks, is_leaf=lambda x: is_block(x) or x is None)

def seq_flatten(blocks: BlocksTree) -> Blocks:
    """Flatten nested blocks into a single list of books and drop Nones
   
    Parameters
    ----------
    blocks : BlocksTree
        a tree of blocks
        
    Returns
    -------
    Blocks
        a list of blocks where the nested blocks are flattened and Nones are dropped
    """
    return tu.tree_leaves(blocks, is_leaf=is_block)

def seq_unflatten(treedef: tu.PyTreeDef, blocks: Blocks) -> BlocksTree:
    """unflatten a list of blocks into a tree of blocks according to a tree structure

    Parameters
    ----------
    treedef : tu.PyTreeDef
        the tree structure of the tree of blocks
        
    Returns
    -------
    BlocksTree
        a tree of blocks where the nested blocks are unflattened 
    """
    return tu.tree_unflatten(treedef, blocks)

def seq_assert_same_structure(*trees: BlocksTree) -> None:
    """check if multiple trees have the same structure, i.e. the same tree
    
    Parameters
    ----------
    trees : BlocksTree
        a sequence of trees of blocks
    
    Raises
    ------
    AssertionError
        if the trees don't have the same structure
    """
    treedefs = [seq_treedef(t, include_none=True) for t in trees]
    assert all(t1 == t2 for t1, t2 in zip(treedefs, treedefs[1:])), "Trees have different structure"

def seq_filter(op: Callable[[Block], bool], blocks: BlocksTree) -> BlocksTree:
    """filter a tree of blocks according to a predicate, which is specified
    through a function that takes a block as input and returns a boolean.
    The unmatched values will be left as None.
    
    Parameters
    ----------
    op : Callable[[Block], bool]
        the predicate function
        
    Returns
    -------
    BlocksTree
        a tree of blocks where the blocks that don't satisfy the predicate are
        replaced by None
    
    """
    return tu.tree_map(lambda b: None if not op(b) else b, blocks, is_leaf=is_block)

def seq_filter_out(op: Callable[[Block], bool], blocks: BlocksTree) -> BlocksTree:
    """filter a tree of blocks according to a predicate, which is specified
    through a function that takes a block as input and returns a boolean.
    The matched values will be left as None (which is opposite to seq_filter).
    
    Parameters
    ----------
    op : Callable[[Block], bool]
        the predicate function
    
    Returns
    -------
    BlocksTree
        a tree of blocks where the blocks that satisfy the predicate are
        replaced by None
    
    """
    return tu.tree_map(lambda b: None if op(b) else b, blocks, is_leaf=is_block)

def seq_map(op, *blocks: BlocksTree):
    """map a function over multiple trees of blocks. The trees must have the
    same structure.
    
    Parameters
    ----------
    op : Callable[[*Block], Any]
        the function to map
    *blocks : BlocksTree 
        a sequence of trees of blocks
    
    Returns
    -------
    a tree of mapped values with the same structure
    """
    return tu.tree_map(op, *blocks, is_leaf=is_block)

def seq_map_with_path(op, *blocks: BlocksTree) -> List[Any]:
    """map a function over multiple trees of blocks. The trees must have the
    same structure. The function takes the path as well as the block as input,
    i.e., op = lambda path, block: ...

    Parameters
    ----------
    op : Callable[[path, *Block], Any]
        the function to map over the trees
    *blocks : BlocksTree
        a sequence of trees of blocks

    Returns
    -------
    a tree of mapped values with the same structure
    """
    return tu.tree_map_with_path(op, *blocks, is_leaf=is_block)

def seq_map_when(op_when: Callable[[Block], bool], op: Callable[[Block], Any], blocks: BlocksTree) -> List[Any]:
    """map a function over a tree of blocks when a predicate is satisfied,
    which is specified through a function that takes a block as input and
    returns a boolean. The unmatched values will be left untouched
    
    Parameters
    ----------
    op_when : Callable[[Block], bool]
        the predicate function
    op : Callable[[Block], Any]
        the function to map
    blocks : BlocksTree
        a tree of blocks
    
    Returns
    -------
    a tree of mapped values with the same structure
    """
    return tu.tree_map(lambda b: op(b) if op_when(b) else b, blocks, is_leaf=is_block)

def seq_replace_block(blocks: BlocksTree, source: Block, target: Block) -> BlocksTree:
    """replace a block in a tree of blocks with another block.
    
    Parameters
    ----------
    blocks : BlocksTree
        a tree of blocks
    source : Block
        the block to be replaced
    target : Block
        the block to replace with

    Returns
    -------
    BlocksTree
        a tree of blocks where the source block is replaced by the target block
    """
    return seq_map_when(lambda b: b == source, lambda _: target, blocks)

def seq_trim(blocks: BlocksTree, t0: dt.datetime, t1: dt.datetime) -> BlocksTree:
    """trim a tree of blocks to a time range
    
    Parameters
    ----------
    blocks : BlocksTree
        a tree of blocks
    t0 : dt.datetime
        the start time of the trimmed blocks
    t1 : dt.datetime
        the end time of the trimmed blocks
    
    Returns
    -------
    BlocksTree
        a tree of blocks where the blocks are trimmed to the time range
    """
    return seq_map(lambda b: b.trim(t0, t1), blocks)

def seq_partition(op, blocks: BlocksTree) -> List[Any]:
    """partition a blockstree into two trees, one for blocks that satisfy the predicate,
    which is specified through a function that takes a block as input and returns
    a boolean, and the second return is for blocks that don't match the predicate. Unmatched
    values will be left as None.
    
    Parameters
    ----------
    op : Callable[[Block], bool]
        the predicate function
    blocks : BlocksTree
        a tree of blocks
    
    Returns
    -------
    matched : BlocksTree
        a tree of blocks where the blocks satisfy the predicate
    unmatched : BlocksTree
        a tree of blocks where the blocks don't satisfy the predicate
    """
    import equinox
    filter_spec = tu.tree_map(op, blocks, is_leaf=is_block)
    return equinox.partition(blocks, filter_spec)

def seq_partition_with_path(op, blocks: BlocksTree, **kwargs) -> List[Any]:
    """partition a blockstree into two trees, one for blocks that satisfy the predicate,
    which is specified through a function that takes a path and block as input and returns
    a boolean, and the second return is for blocks that don't match the predicate. Unmatched
    values will be left as None.
    
    Parameters
    ----------
    op : Callable[[path, Block], bool]
        the predicate function
        
    Returns
    -------
    matched : BlocksTree
        a tree of blocks where the blocks satisfy the predicate
    unmatched : BlocksTree
        a tree of blocks where the blocks don't satisfy the predicate
    """
    import equinox
    filter_spec = tu.tree_map_with_path(op, blocks, is_leaf=is_block)
    return equinox.partition(blocks, filter_spec, **kwargs)

def seq_partition_with_query(query, blocks: BlocksTree):
    """partition a blockstree into two trees, one for blocks that satisfy the query,
    which is specified through a query string, and the second return is for blocks that
    don't match the query. See utils.match_query for details of query match syntax.

    Parameters
    ----------
    query : str
        the query string
    blocks : BlocksTree
        a tree of blocks
    
    Returns
    -------
    matched : BlocksTree
        a tree of blocks where the blocks satisfy the query
    unmatched : BlocksTree
        a tree of blocks where the blocks don't satisfy the query
    """
    return seq_partition_with_path(lambda path, block: utils.match_query(path, query), blocks)

def seq_combine(*blocks: BlocksTree) -> BlocksTree:
    """combine blocks from multiple trees into a single tree, where the blocks are
    combined in a list. The trees must have the same structure. This is meant to work
    with trees splitted by seq_partition. No merging is done.
    
    Parameters
    ----------
    *blocks : BlocksTree
        a sequence of trees of blocks
     
    Returns
    -------
    BlocksTree
        a tree of blocks where the blocks are combined in a list
    """
    import equinox
    seq_assert_same_structure(*blocks)
    return equinox.combine(*blocks, is_leaf=is_block)

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
class GreenRule(BlocksTransformation, ABC):
    """GreenRule preserves trees. A check is explicitly made to ensure
    that the input and output are both trees."""
    def __call__(self, blocks: BlocksTree) -> BlocksTree:
        out = self.apply(blocks)
        # did we destroy nestedness?
        if not seq_is_nested(out) and seq_is_nested(blocks):
            raise RuntimeError("GreenRule should preserves trees, not destroying trees!")
        return out

@dataclass(frozen=True)
class ConstrainedRule(GreenRule):
    """ConstrainedRule applies a rule to a subset of blocks. Here
    constraint is a fnmatch pattern that matches to the `key` of a
    block. This is implemented by first partitioning the tree into
    one part that matches the constraint and one part that doesn't,
    and then applying the rule to the matching part before combining
    the two parts back together.

    Parameters
    ----------
    rule : Rule. the rule to apply to the matching blocks
    constraint : str. fnmatch pattern that matches to the `key` of a block
    """
    rule: Rule
    constraint: str
    def apply(self, blocks: BlocksTree) -> BlocksTree:
        matched, unmatched = seq_partition_with_query(self.constraint, blocks)
        return seq_combine(self.rule(matched), unmatched)


@dataclass(frozen=True)
class MappableRule(GreenRule, ABC):
    """MappableRule applies the same rule to all blocks in a tree. One needs
    to implement the `apply_block` method to define how the rule is applied
    to a single block."""
    def apply(self, blocks: BlocksTree) -> BlocksTree:
        return seq_map(self.apply_block, blocks)
    @abstractmethod
    def apply_block(self, block) -> Blocks: ...

@dataclass(frozen=True)
class Policy(BlocksTransformation, ABC):
    """apply: apply policy to a tree of blocks"""
    # initialize a tree of blocks
    @abstractmethod
    def init_seqs(self) -> BlocksTree: ...
    @abstractmethod
    def apply(self, blocks: BlocksTree) -> Blocks: ...

@dataclass(frozen=True)
class BasePolicy(Policy, ABC):
    """we split the policy into two parts: transform and merge where
    transform are the part that preserves nested structure and merge
    is the part that flattens the nested structure into a single
    sequence. This is mostly for visualization purposes, so that we
    preserve the nested structure for the user to see, but we can
    also flatten the structure for the scheduler to consume."""

    def transform(self, blocks: BlocksTree) -> BlocksTree:
        return blocks

    def merge(self, blocks: BlocksTree) -> Blocks:
        return blocks

    def apply(self, blocks: BlocksTree) -> Blocks:
        """main interface"""
        blocks = self.transform(blocks)
        blocks = self.merge(blocks)
        return blocks

    @abstractmethod
    def seq2cmd(self, seq: Blocks) -> str: ...

# ===============================
# Others convenience types alias
# ===============================

Arr = np.ndarray
