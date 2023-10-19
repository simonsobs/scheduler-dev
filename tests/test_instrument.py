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
