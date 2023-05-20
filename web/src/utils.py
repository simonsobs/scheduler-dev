from schedlib import core, utils as u
from typing import Dict
import jax.tree_util as tu
from functools import partial

path2key = lambda path: ".".join([str(p.key) for p in path])

def update_with_path(data, path, value):
    if len(path) == 0: return data
    for i in range(0, len(path)-1):
        key = path2key(path[:i+1])
        if path[i].key not in data: data[key] = {}
        data = data[key]
    data[path2key(path)] = value
    return data

def groups_unfold(tree, is_leaf) -> Dict[str, core.Blocks]:
    res = {}
    tu.tree_map_with_path(lambda path, x: update_with_path(res, path, path2key(path)), tree, is_leaf=is_leaf)
    return res

def make_group(tree, is_leaf=lambda x: isinstance(x, str)):
    groups = []
    for key in tree:
        v = tree[key]
        if is_leaf(v):
            groups.append({'id': key, 'content': key.split('.')[-1]})
        else:
            groups.append({'id': key, 'content': key.split('.')[-1], 'nestedGroups': list(v.keys())})
            groups.extend(make_group(v, is_leaf=is_leaf))
    return groups

def tree_unfold(tree, is_leaf):
    res = {}
    tu.tree_map_with_path(lambda path, x: res.update({path2key(path): x}), tree, is_leaf=is_leaf)
    return res

def block2dict(block, group=None):
    res = {
        'id': hash(block),
        'content': block.name,
        'start': block.t0.isoformat(),
        'end': block.t1.isoformat(),
    }
    if group is not None: res['group'] = group
    return res

def seq2visdata_flat(seqs):
    # make group
    is_list = lambda x: isinstance(x, list)
    groups = []
    tu.tree_map_with_path(lambda path, x: groups.append({'id': path2key(path), 'content': path2key(path)}), seqs, is_leaf=is_list)
    # make items
    seqs = tu.tree_leaves(
        tu.tree_map_with_path(
            lambda path, x: core.seq_map(partial(block2dict, group=path2key(path)), core.seq_sort(x, flatten=True)),
            seqs, is_leaf=is_list),
        is_leaf=lambda x: 'id' in x)
    return seqs, groups

def seq2visdata_nested(seqs):
    # make group
    is_list = lambda x: isinstance(x, list)
    unfolded_groups = groups_unfold(seqs, is_leaf=is_list)
    groups = make_group(unfolded_groups)
    # make items
    seqs = tu.tree_leaves(
        tu.tree_map_with_path(
            lambda path, x: core.seq_map(partial(block2dict, group=path2key(path)), core.seq_sort(x, flatten=True)),
            seqs, is_leaf=is_list),
        is_leaf=lambda x: 'id' in x)
    return seqs, groups
