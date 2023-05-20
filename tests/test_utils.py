import pytest
from datetime import datetime, timezone
from schedlib.utils import *
import os.path as op

def test_str2ctime_valid_time_string():
    time_str = "2023-05-14 12:34:56"
    expected_output = 1684067696
    assert str2ctime(time_str) == expected_output

def test_str2ctime_invalid_time_string():
    time_str = "abc"
    with pytest.raises(ValueError):
        str2ctime(time_str)

def test_str2datetime_valid_time_string():
    time_str = "2023-05-14 12:34:56"
    expected_output = datetime(2023, 5, 14, 12, 34, 56, tzinfo=timezone.utc)
    assert str2datetime(time_str) == expected_output

def test_mask2ranges_empty_mask():
    mask = np.array([])
    expected_output = np.empty((0, 2), dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_single_true():
    mask = np.array([False, False, True, False, False])
    expected_output = np.array([[2, 3]], dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_single_false():
    mask = np.array([True, True, False, True, True])
    expected_output = np.array([[0, 2], [3, 5]], dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_multiple_ranges():
    mask = np.array([True, False, False, True, True, False, True, False])
    expected_output = np.array([[0, 1], [3, 5], [6, 7]], dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_all_true():
    mask = np.array([True, True, True, True])
    expected_output = np.array([[0, 4]], dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_all_false():
    mask = np.array([False, False, False, False])
    expected_output = np.empty((0, 2), dtype=int)
    assert np.array_equal(mask2ranges(mask), expected_output)

def test_mask2ranges_digest():
    mask = np.array([False, False, True, True, False, False, True, True, True, False, False])
    for i_left, i_right in mask2ranges(mask):
        assert np.all(mask[i_left:i_right])

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "data/schedule_sat.txt")
    seq = parse_sequence_from_toast(ifile)
    assert len(seq) == 20

def test_ranges_pad():
    mask = np.array([False, False, True, True, False, False, False, True, True, True, False, False])
    ranges = mask2ranges(mask)
    ranges = ranges_pad(ranges, 1, len(mask))
    assert np.alltrue(ranges == [[1, 5], [6, 11]])

    ranges = mask2ranges(mask)
    ranges = ranges_pad(ranges, 3, len(mask))
    assert np.alltrue(ranges == [[0, 12]])

def test_pngkey():
    key1 = PRNGKey(42)
    key2 = PRNGKey(41)
    # repeated calls should return the same value and not change
    # the result of other calls
    with key1.set_state():
        assert np.isclose(np.random.uniform(0, 1), 0.374540)
    with key2.set_state():
        assert np.isclose(np.random.uniform(0, 1), 0.250923)
    with key1.set_state():
        assert np.isclose(np.random.uniform(0, 1), 0.374540)
    with key2.set_state():
        assert np.isclose(np.random.uniform(0, 1), 0.250923)

def test_uniform():
    key = PRNGKey(42)
    assert np.isclose(uniform(key, 0, 1), 0.374540)
    assert np.isclose(uniform(key, 0, 1), 0.374540)
    new_key, _ = key.split()
    assert np.isclose(uniform(new_key, 0, 1), 0.500341)
    assert np.isclose(uniform(key, 0, 1), 0.374540)

def test_daily_static_key():
    key = daily_static_key(datetime(2020, 1, 1))
    v = uniform(key, 0, 1)
    assert np.isclose(v, 0.373878)

def test_interp_extra():
    x = np.array([1, 2, 3, 4, 5])[::-1]
    y = np.array([1, 2, 3, 4, 5])[::-1]
    c = np.array([1, 2, 4, 5, 6])
    y_new = interp_extra(c, x, y)
    assert np.allclose(y_new, np.array([1, 2, 4, 5, 6]))