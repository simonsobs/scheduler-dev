from . import core
from chex import dataclass
from jax import tree_util as tu
from typing import List, TypeVar, Union, Dict
import numpy as np
from functools import reduce

@dataclass(frozen=True)
class ScanBlock(core.NamedBlock):
    az: float     # deg
    alt: float    # deg
    throw: float  # deg
    patch: str

@dataclass(frozen=True)
class TrackingBlock(core.NamedBlock):
    t_start: core.Arr[float]
    obs_length: core.Arr[float]
    az_bore: core.Arr[float]
    alt_bore: core.Arr[float]
    az_throw: core.Arr[float]

@dataclass(frozen=True)
class IVBlock(core.NamedBlock): pass



# dummy type variable for readability
Spec = TypeVar('Spec')
SpecsTree = Dict[str, Union[Spec, "SpecsTree"]]

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
def get_spec(specs: SpecsTree, query: List[str], merge=True) -> Union[Spec, SpecsTree]:
    """return a reduced spec (if merge=True) from all specs that match
    one of the queries. return all matches if merge=False"""
    is_leaf = lambda x: isinstance(x, dict) and 'bounds_x' in x
    match_p = lambda key: any([p in key for p in query])
    path2key = lambda path: ".".join([str(p.key) for p in path])
    def reduce_fn(l, r):
        res = {}
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

def get_bounds_x_tilted(bounds_x: List[float], bounds_y: List[float], phi_tilt: Union[float, core.Arr[float]], shape: str):
    """get the effective bounds of the x-axis of the spec when covering a tilted patch"""
    assert shape in ['ellipse']  # more to implement
    if shape == 'ellipse':
        a = (bounds_x[1] - bounds_x[0])/2
        b = (bounds_y[1] - bounds_y[0])/2
        w_proj = np.sqrt(a**2 * np.cos(phi_tilt)**2 + b**2 * np.sin(phi_tilt)**2)
        return np.array([-w_proj, w_proj]) + (bounds_x[0] + bounds_x[1])/2
    else:
        raise NotImplementedError
