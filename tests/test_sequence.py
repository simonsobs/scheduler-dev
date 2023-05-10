import os, os.path as op
from schedlib.core import ScanBlock, Block, Sequence, block_split, seq_concat, seq_sort, seq_has_overlap, seq_is_sorted, seq_assert_sorted, seq_assert_no_overlap, blocks_sort
from schedlib.parse import parse_sequence_from_toast
import datetime as dt

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "data/schedule_sat.txt")
    seq = parse_sequence_from_toast(ifile)
    assert len(seq.blocks) == 20

def test_basic():
    # Test case 1
    block1 = ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")
    seq1 = block_split(block1, dt.datetime(2023, 5, 9, 12, 0, 0))
    assert seq1.blocks == [block1]

    # Test case 2
    block2 = ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")
    seq2 = block_split(block2, dt.datetime(2023, 5, 11, 12, 0, 0))
    assert seq2.blocks == [block2]

    # Test case 3
    block3 = ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")
    seq3 = block_split(block3, dt.datetime(2023, 5, 9, 12, 0, 0, 1))
    assert seq3.blocks == [ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 9, 12, 0, 0, 1), 45.0, 180.0, 10.0, "red"),
                           ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0, 1), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")]

    # Test case 4
    block4 = ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")
    seq4 = block_split(block4, dt.datetime(2023, 5, 10, 0, 0, 0))
    assert seq4.blocks == [ScanBlock(dt.datetime(2023, 5, 9, 12, 0, 0), dt.datetime(2023, 5, 10, 0, 0, 0), 45.0, 180.0, 10.0, "red"),                        
                           ScanBlock(dt.datetime(2023, 5, 10, 0, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0), 45.0, 180.0, 10.0, "red")]

def test_block_split():
    b = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = block_split(b, dt.datetime(2023, 5, 10, 9, 0, 0))
    s2 = block_split(b, dt.datetime(2023, 5, 10, 11, 0, 0))
    s3 = block_split(b, dt.datetime(2023, 5, 10, 13, 0, 0))
    assert len(s1.blocks) == 1 and s1.blocks[0] == b
    assert s2.blocks[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2.blocks[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s2.blocks) == 2 and s2.blocks[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2.blocks[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s3.blocks) == 1 and s3.blocks[0] == b

def test_blocks_sort():
    b1 = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    b2 = Block(dt.datetime(2023, 5, 10, 8, 0, 0), dt.datetime(2023, 5, 10, 9, 0, 0))
    b3 = Block(dt.datetime(2023, 5, 10, 9, 30, 0), dt.datetime(2023, 5, 10, 11, 0, 0))
    assert blocks_sort([b1, b2, b3]) == [b2, b3, b1]

def test_seq_concat():
    b1 = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    b2 = Block(dt.datetime(2023, 5, 10, 12, 0, 0), dt.datetime(2023, 5, 10, 14, 0, 0))
    s1 = Sequence.from_blocks([b1])
    s2 = Sequence.from_blocks([b2])
    s3 = seq_concat(s1, s2)
    assert len(s3.blocks) == 2 and s3.blocks[0] == b1 and s3.blocks[1] == b2

def test_seq_sort():
    b1 = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    b2 = Block(dt.datetime(2023, 5, 10, 8, 0, 0), dt.datetime(2023, 5, 10, 9, 0, 0))
   
def test_seq_has_overlap():
    # Test with no overlap
    b1 = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = Block(dt.datetime(2023, 5, 10, 11, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = Sequence.from_blocks([b1, b2])
    assert not seq_has_overlap(s1)

    # Test with overlap
    b3 = Block(dt.datetime(2023, 5, 10, 11, 30, 0), dt.datetime(2023, 5, 10, 12, 30, 0))
    s2 = Sequence.from_blocks([b1, b2, b3])
    assert seq_has_overlap(s2)

def test_seq_is_sorted():
    # Test with sorted sequence
    b1 = Block(dt.datetime(2023, 5, 10, 10, 0, 0), dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = Block(dt.datetime(2023, 5, 10, 11, 0, 0), dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = Sequence.from_blocks([b1, b2])
    assert seq_is_sorted(s1)

    # Test with unsorted sequence
    b3 = Block(dt.datetime(2023, 5, 10, 9, 0, 0), dt.datetime(2023, 5, 10, 10, 0, 0))
    s2 = Sequence.from_blocks([b2, b3, b1])
    assert not seq_is_sorted(s2)

    # Test with single block sequence
    s3 = Sequence.from_blocks([b1])
    assert seq_is_sorted(s3)