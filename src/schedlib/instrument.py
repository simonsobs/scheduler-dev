from . import core
from chex import dataclass
from jax import tree_util as tu
from typing import List, TypeVar
import numpy as np
from functools import reduce

@dataclass(frozen=True)
class ScanBlock(core.NamedBlock):
    az: float     # deg
    alt: float    # deg
    throw: float  # deg
    patch: str

@dataclass(frozen=True)
class IVBlock(core.NamedBlock): pass


# dummy type variable for readability
SpecsTree = TypeVar('SpecsTree')

# SpecsTree can be an arbitrarily nested dict, with each leaf node being a dict
# with the following keys:
#  {
#       'bounds_x: [-1.0, 1.0],
#       'bounds_y: [-1.0, 1.0],
#  }
# To query a specific set of spec, use get_spec with a list of strings as query, where
# each string will be matched with the dot-separated path of a leaf node. Leaves that
# match *any* of the provided queries will be collected and reduced to a single leaf
# node.
def get_spec(specs: SpecsTree, query: List[str], merge=True):
    """return a reduced spec (if merge=True) from all specs that match
    one of the queries. return all matches if merge=False"""
    is_leaf = lambda x: isinstance(x, dict) and 'bounds_x' in x
    match_p = lambda key: any([p in key for p in query])
    path2key = lambda path: ".".join([str(p.key) for p in path])
    def reduce_fn(l, r):
        for k in ['bounds_x', 'bounds_y']:
            res[k] = [min(l[k][0], r[k][0]), max(l[k][1], r[k][1])]
        return res
    all_matches = tu.tree_leaves(
        tu.tree_map_with_path(lambda path, x: x if match_p(path2key(path)) else None, specs, is_leaf=is_leaf), 
        is_leaf=is_leaf
    )  # None is not a leaf, so it will be filtered out
    if not merge: return all_matches
    if len(all_matches) == 0: return {}
    return reduce(reduce_fn, all_matches[1:], all_matches[0])
