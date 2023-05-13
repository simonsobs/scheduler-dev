import os.path as op
import schedlib.core as sl
from schedlib.parse import parse_sequence_from_toast
import datetime as dt

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "data/schedule_sat.txt")
    seq = parse_sequence_from_toast(ifile)
    assert len(seq) == 20

def test_basic():
    # Test case 1
    block1 = sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq1 = sl.block_split(block1, dt.datetime(2023, 5, 9, 12, 0, 0))
    assert seq1== [block1]

    # Test case 2
    block2 = sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq2 = sl.block_split(block2, dt.datetime(2023, 5, 11, 12, 0, 0))
    assert seq2== [block2]

    # Test case 3
    block3 = sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq3 = sl.block_split(block3, dt.datetime(2023, 5, 9, 12, 0, 0, 1))
    assert seq3 == [sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 9, 12, 0, 0, 1)),
                           sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0, 1), t1=dt.datetime(2023, 5, 10, 12, 0, 0))]

    # Test case 4
    block4 = sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq4 = sl.block_split(block4, dt.datetime(2023, 5, 10, 0, 0, 0))
    assert seq4 == [sl.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 0, 0, 0)),                        
                           sl.Block(t0=dt.datetime(2023, 5, 10, 0, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))]

def test_block_split():
    b = sl.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = sl.block_split(b, dt.datetime(2023, 5, 10, 9, 0, 0))
    s2 = sl.block_split(b, dt.datetime(2023, 5, 10, 11, 0, 0))
    s3 = sl.block_split(b, dt.datetime(2023, 5, 10, 13, 0, 0))
    assert len(s1) == 1 and s1[0] == b
    assert s2[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s2) == 2 and s2[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s3) == 1 and s3[0] == b

def test_seq_sort():
    b1 = sl.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    b2 = sl.Block(t0=dt.datetime(2023, 5, 10, 8, 0, 0), t1=dt.datetime(2023, 5, 10, 9, 0, 0))
    b3 = sl.Block(t0=dt.datetime(2023, 5, 10, 9, 30, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    assert sl.seq_sort([b1, b2, b3]) == [b2, b3, b1]

def test_seq_has_overlap():
    # Test with no overlap
    b1 = sl.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = sl.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, b2]
    assert not sl.seq_has_overlap(s1)

    # Test with overlap
    b3 = sl.Block(t0=dt.datetime(2023, 5, 10, 11, 30, 0), t1=dt.datetime(2023, 5, 10, 12, 30, 0))
    s2 = [b1, b2, b3]
    assert sl.seq_has_overlap(s2)

def test_seq_is_sorted():
    # Test with sorted sequence
    b1 = sl.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = sl.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, b2]
    assert sl.seq_is_sorted(s1)

    # Test with unsorted sequence
    b3 = sl.Block(t0=dt.datetime(2023, 5, 10, 9, 0, 0), t1=dt.datetime(2023, 5, 10, 10, 0, 0))
    s2 = [b2, b3, b1]
    assert not sl.seq_is_sorted(s2)

    # Test with single block sequence
    s3 = [b1]
    assert sl.seq_is_sorted(s3)