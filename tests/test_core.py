from schedlib import core
import datetime as dt

def test_block_split():
    # Test case 1
    block1 = core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq1 = core.block_split(block1, dt.datetime(2023, 5, 9, 12, 0, 0))
    assert seq1 == [block1]

    # Test case 2
    block2 = core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq2 = core.block_split(block2, dt.datetime(2023, 5, 11, 12, 0, 0))
    assert seq2 == [block2]

    # Test case 3
    block3 = core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq3 = core.block_split(block3, dt.datetime(2023, 5, 9, 12, 0, 0, 1))
    assert seq3 == [core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 9, 12, 0, 0, 1)),
                    core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0, 1), t1=dt.datetime(2023, 5, 10, 12, 0, 0))]

    # Test case 4
    block4 = core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    seq4 = core.block_split(block4, dt.datetime(2023, 5, 10, 0, 0, 0))
    assert seq4 == [core.Block(t0=dt.datetime(2023, 5, 9, 12, 0, 0), t1=dt.datetime(2023, 5, 10, 0, 0, 0)),                        
                           core.Block(t0=dt.datetime(2023, 5, 10, 0, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))]

    # Test case 5
    b = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = core.block_split(b, dt.datetime(2023, 5, 10, 9, 0, 0))
    s2 = core.block_split(b, dt.datetime(2023, 5, 10, 11, 0, 0))
    s3 = core.block_split(b, dt.datetime(2023, 5, 10, 13, 0, 0))
    assert len(s1) == 1 and s1[0] == b
    assert s2[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s2) == 2 and s2[0].t1 == dt.datetime(2023, 5, 10, 11, 0, 0) and s2[1].t0 == dt.datetime(2023, 5, 10, 11, 0, 0)
    assert len(s3) == 1 and s3[0] == b

def test_block_trim_within_range():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block from 10:30 to 11:30
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 10, 30), dt.datetime(2023, 1, 1, 11, 30))

    # Check if the trimmed block is correct
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 10, 30)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 11, 30)

def test_block_trim_outside_range():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block from 9:00 to 9:30 (outside range)
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 9, 0), dt.datetime(2023, 1, 1, 9, 30))

    # Check if the trimmed block is None
    assert trimmed_block is None

def test_block_trim_partial_overlap():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block from 11:30 to 13:00 (partial overlap)
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 11, 30), dt.datetime(2023, 1, 1, 13, 0))

    # Check if the trimmed block is correct
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 11, 30)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 12, 0)

def test_block_trim_full_overlap():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block from 9:00 to 13:00 (full overlap)
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 9, 0), dt.datetime(2023, 1, 1, 13, 0))

    # Check if the trimmed block is correct
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 10, 0)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 12, 0)

def test_block_trim_no_specified_range():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block without specifying a range (use default values)
    trimmed_block = core.block_trim(block)

    # Check if the trimmed block is the same as the original block
    assert trimmed_block.t0 == block.t0
    assert trimmed_block.t1 == block.t1

def test_block_shift():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Shift the block by 1 hour
    shifted_block = core.block_shift(block, dt.timedelta(hours=1))

    # Check if the shifted block has the correct time range
    assert shifted_block.t0 == dt.datetime(2023, 1, 1, 11, 0)
    assert shifted_block.t1 == dt.datetime(2023, 1, 1, 13, 0)

def test_block_trim_left_to():
    # Create a block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Trim the block to the left up to 11:00
    trimmed_block = core.block_trim_left_to(block, dt.datetime(2023, 1, 1, 11, 0))

    # Check if the trimmed block is correct
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 11, 0)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 12, 0)

    # Trim the block to the left up to 9:00 (no change expected)
    trimmed_block_outside = core.block_trim_left_to(block, dt.datetime(2023, 1, 1, 9, 0))

    # Check if the trimmed block remains the same
    assert trimmed_block_outside.t0 == block.t0
    assert trimmed_block_outside.t1 == block.t1

    # Trim the block to the left up to 9:00 (no change expected)
    trimmed_block_outside = core.block_trim_left_to(block, dt.datetime(2023, 1, 1, 13, 0))

    # Check if the trimmed block is None
    assert trimmed_block_outside is None

def test_block_isa():
    class CustomBlock(core.Block): pass

    # Create a regular block from 10:00 to 12:00
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))

    # Create a custom block (subclass of Block) from 11:00 to 13:00
    custom_block = CustomBlock(t0=dt.datetime(2023, 1, 1, 11, 0), t1=dt.datetime(2023, 1, 1, 13, 0))

    # Define an "is_regular_block" function using block_isa with Block type
    is_regular_block = core.block_isa(core.Block)

    # Check if the regular block is recognized as a Block
    assert is_regular_block(block) is True

    # Check if the custom block is recognized as a Block
    assert is_regular_block(custom_block) is True

    # Define an "is_custom_block" function using block_isa with CustomBlock type
    is_custom_block = core.block_isa(CustomBlock)

    # Check if the regular block is recognized as a CustomBlock
    assert is_custom_block(block) is False

    # Check if the custom block is recognized as a CustomBlock
    assert is_custom_block(custom_block) is True

def test_seq_has_nested():
    # Create a list of blocks without nested blocks
    blocks_without_nested = [
        core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 11, 0)),
        core.Block(t0=dt.datetime(2023, 1, 1, 12, 0), t1=dt.datetime(2023, 1, 1, 13, 0)),
        core.Block(t0=dt.datetime(2023, 1, 1, 14, 0), t1=dt.datetime(2023, 1, 1, 15, 0))
    ]
    # Check if seq_has_nested returns False for blocks without nested blocks
    assert core.seq_is_nested(blocks_without_nested) is False

    # Create a list of blocks with a nested block
    blocks_with_nested = [
        core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 11, 0)),
        [core.Block(t0=dt.datetime(2023, 1, 1, 12, 0), t1=dt.datetime(2023, 1, 1, 13, 0))],
        core.Block(t0=dt.datetime(2023, 1, 1, 14, 0), t1=dt.datetime(2023, 1, 1, 15, 0))
    ]

    # Check if seq_has_nested returns True for blocks with a nested block
    assert core.seq_is_nested(blocks_with_nested) is True

def test_seq_flatten():
    # Create a nested list of blocks
    nested_blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 11, 0)),
        [
            [core.Block(t0=dt.datetime(2023, 1, 1, 12, 0), t1=dt.datetime(2023, 1, 1, 13, 0)),
             None,
             []],
            None,
            core.Block(t0=dt.datetime(2023, 1, 1, 14, 0), t1=dt.datetime(2023, 1, 1, 15, 0))
        ],
        [],
        core.Block(t0=dt.datetime(2023, 1, 1, 16, 0), t1=dt.datetime(2023, 1, 1, 17, 0))
    ]

    # Flatten the nested list of blocks using seq_flatten
    flattened_blocks = core.seq_flatten(nested_blocks)

    # Check if the flattened blocks match the expected result
    expected_blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 11, 0)),
        core.Block(t0=dt.datetime(2023, 1, 1, 12, 0), t1=dt.datetime(2023, 1, 1, 13, 0)),
        core.Block(t0=dt.datetime(2023, 1, 1, 14, 0), t1=dt.datetime(2023, 1, 1, 15, 0)),
        core.Block(t0=dt.datetime(2023, 1, 1, 16, 0), t1=dt.datetime(2023, 1, 1, 17, 0))
    ]
    assert flattened_blocks == expected_blocks

def test_seq_sort():
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 8, 0, 0), t1=dt.datetime(2023, 5, 10, 9, 0, 0))
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 9, 30, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    assert core.seq_sort([b1, None, [b2, None, b3]]) == [b2, b3, b1]

def test_seq_has_overlap():
    # Test with no overlap
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, b2]
    assert not core.seq_has_overlap(s1)

    # Test with overlap
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 30, 0), t1=dt.datetime(2023, 5, 10, 12, 30, 0))
    s2 = [b1, b2, b3]
    assert core.seq_has_overlap(s2)

def test_seq_is_sorted():
    # Test with sorted sequence
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, b2]
    assert core.seq_is_sorted(s1)

    # Test with unsorted sequence
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 9, 0, 0), t1=dt.datetime(2023, 5, 10, 10, 0, 0))
    s2 = [b2, b3, b1]
    assert not core.seq_is_sorted(s2)

    # Test with single block sequence
    s3 = [b1]
    assert core.seq_is_sorted(s3)