import numpy as np
import os.path as op
import schedlib.instrument as inst

def test_get_spec():
    specs = {
        'platform1': {
            'wafer1': {
                'bounds_x': [-1.0, 1.0],
                'bounds_y': [-1.0, 1.0],
            },
            'wafer2': {
                'bounds_x': [-2.0, 1.0],
                'bounds_y': [-2.0, 1.0],
            },
        }
    }
    spec = inst.get_spec(specs, ["platform1"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = inst.get_spec(specs, ["wafer1"])
    assert spec == {
        'bounds_x': [-1.0, 1.0],
        'bounds_y': [-1.0, 1.0],
    }
    spec = inst.get_spec(specs, ["platform1.wafer2"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = inst.get_spec(specs, ["wafer"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = inst.get_spec(specs, ["wafer3"])
    assert spec == {}

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "data/schedule_test.txt")
    seq = inst.parse_sequence_from_toast(ifile)
    print(seq)
    assert len(seq) == 17

def test_array_info():
    geometries = {
        'w11': {
            'center': [0, 0],
            'radius': 1.0,
        },
        'w22': {
            'center': [1, 1],
            'radius': 1.0,
        }
    }
    array_info1 = inst.make_circular_cover(*geometries['w11']['center'], geometries['w11']['radius'])
    array_info2 = inst.make_circular_cover(*geometries['w22']['center'], geometries['w22']['radius'])
    assert array_info1['cover'].shape == (2, 50)
    query = "w11,w22"
    array_info = inst.array_info_from_query(geometries, query)
    assert array_info['cover'].shape == (2, 100)
    assert np.allclose(array_info['center'], np.mean([array_info1['center'], array_info2['center']], axis=0))
    assert np.allclose(array_info['cover'], np.concatenate([array_info1['cover'], array_info2['cover']], axis=1))

    query = "w1*"
    array_info = inst.array_info_from_query(geometries, query)
    assert array_info['cover'].shape == (2, 50)
    assert np.allclose(array_info['center'], array_info1['center'])
    assert np.allclose(array_info['cover'], array_info1['cover'])

    query = "*2"
    array_info = inst.array_info_from_query(geometries, query)
    assert array_info['cover'].shape == (2, 50)
    assert np.allclose(array_info['center'], array_info2['center'])
    assert np.allclose(array_info['cover'], array_info2['cover'])

    query = "*2,*1"
    array_info = inst.array_info_from_query(geometries, query)
    assert array_info['cover'].shape == (2, 100)
    assert np.allclose(array_info['center'], np.mean([array_info1['center'], array_info2['center']], axis=0))
    assert np.allclose(array_info['cover'], np.concatenate([array_info1['cover'], array_info2['cover']], axis=1))