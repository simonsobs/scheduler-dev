from schedlib.instrument import get_spec

def test_get_spec():
    specs = {
        'platform1': {
            'wafer1': {
                'pos': [0.0, 0.0],
                'bounds_x': [-1.0, 1.0],
                'bounds_y': [-1.0, 1.0],
            },
            'wafer2': {
                'pos': [2.0, 2.0],
                'bounds_x': [-2.0, 1.0],
                'bounds_y': [-2.0, 1.0],
            },
        }
    }
    spec = get_spec(specs, ["platform1"])
    assert spec == {
        'pos': [1.0, 1.0],
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer1"])
    assert spec == {
        'pos': [0.0, 0.0],
        'bounds_x': [-1.0, 1.0],
        'bounds_y': [-1.0, 1.0],
    }
    spec = get_spec(specs, ["platform1.wafer2"])
    assert spec == {
        'pos': [2, 2],
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer"])
    assert spec == {
        'pos': [1.0, 1.0],
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer3"])
    assert spec == {}
