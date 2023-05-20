from schedlib.instrument import get_spec

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
    spec = get_spec(specs, ["platform1"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer1"])
    assert spec == {
        'bounds_x': [-1.0, 1.0],
        'bounds_y': [-1.0, 1.0],
    }
    spec = get_spec(specs, ["platform1.wafer2"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer"])
    assert spec == {
        'bounds_x': [-2.0, 1.0],
        'bounds_y': [-2.0, 1.0],
    }
    spec = get_spec(specs, ["wafer3"])
    assert spec == {}
