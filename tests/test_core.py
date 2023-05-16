from schedlib import core
import datetime as dt
import pytest

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

def test_block_trim():
    # within range
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 10, 30), dt.datetime(2023, 1, 1, 11, 30))
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 10, 30)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 11, 30)

    # outside range
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 9, 0), dt.datetime(2023, 1, 1, 9, 30))
    assert trimmed_block is None

    # partial overlap
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 11, 30), dt.datetime(2023, 1, 1, 13, 0))
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 11, 30)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 12, 0)

    # full overlap
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))
    trimmed_block = core.block_trim(block, dt.datetime(2023, 1, 1, 9, 0), dt.datetime(2023, 1, 1, 13, 0))
    assert trimmed_block.t0 == dt.datetime(2023, 1, 1, 10, 0)
    assert trimmed_block.t1 == dt.datetime(2023, 1, 1, 12, 0)

    # no specification
    block = core.Block(t0=dt.datetime(2023, 1, 1, 10, 0), t1=dt.datetime(2023, 1, 1, 12, 0))
    trimmed_block = core.block_trim(block)
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

def test_seq_is_nested():
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
    # cannot sort nested blocks without flatten=True
    with pytest.raises(ValueError):
        core.seq_sort([b1, None, [b2, None, b3]])
    assert core.seq_sort([b1, None, [b2, None, b3]], flatten=True) == [b2, b3, b1]

def test_seq_has_overlap():
    # Test with no overlap
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, [None, b2]]
    assert not core.seq_has_overlap(s1)

    # Test with overlap
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 30, 0), t1=dt.datetime(2023, 5, 10, 12, 30, 0))
    s2 = [b1, [b2, b3]]
    assert core.seq_has_overlap(s2)

def test_seq_is_sorted():
    # Test with sorted sequence
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    s1 = [b1, None, b2]
    assert core.seq_is_sorted(s1)

    # Test with unsorted sequence
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 9, 0, 0), t1=dt.datetime(2023, 5, 10, 10, 0, 0))
    s2 = [b2, [b3, [b1, None]]] # order and nesting should not matter
    assert not core.seq_is_sorted(s2)

    # Test with single block sequence
    s3 = [b1]
    assert core.seq_is_sorted(s3)

def test_has_overlap():
    # Test with no overlap
    b1 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 0, 0), t1=dt.datetime(2023, 5, 10, 11, 0, 0))
    b2 = core.Block(t0=dt.datetime(2023, 5, 10, 11, 0, 0), t1=dt.datetime(2023, 5, 10, 12, 0, 0))
    assert not core.seq_has_overlap([b1, b2])

    # Test with overlap
    b3 = core.Block(t0=dt.datetime(2023, 5, 10, 10, 30, 0), t1=dt.datetime(2023, 5, 10, 11, 30, 0))
    assert core.seq_has_overlap([b3, [b1, None]]) # order and nesting should not matter

def test_seq_filter():
    # Test case 1: Filtering blocks where t0 is before a specific date
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        None,
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
         None, core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6))],
        core.Block(t0=dt.datetime(2023, 1, 6), t1=dt.datetime(2023, 1, 7)),
    ]
    filtered_blocks = core.seq_filter(lambda b: b.t0 < dt.datetime(2023, 1, 4), blocks)
    assert len(filtered_blocks) == 4
    assert filtered_blocks == [
        blocks[0], 
        None, 
        [blocks[2][0], None, None],
        None
    ]

    # Test case 2: Filtering blocks where t1 is after a specific date
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4))],
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
    ]
    filtered_blocks = core.seq_filter(lambda b: b.t1 > dt.datetime(2023, 1, 3), blocks)
    assert len(filtered_blocks) == 3
    assert filtered_blocks == [
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4))],
        None
    ]

    # Test case 3: Filtering blocks where t0 and t1 are the same
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 1)),
        core.Block(t0=dt.datetime(2023, 1, 2), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 3)),
    ]
    filtered_blocks = core.seq_filter(lambda b: b.t0 == b.t1, blocks)
    assert len(filtered_blocks) == 3
    
def test_seq_map():
    # case 1: nested blocks with Nones: should preserve nesting and Nones
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 5)), [None]],
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
        None,
    ]
    durations = core.seq_map(lambda b: (b.t1 - b.t0).total_seconds(), blocks)
    assert len(durations) == 4
    assert durations == [
        86400, [86400*2, [None]], 86400, None
    ]

    # case 2: blocks with [], should preserve nesting and []
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        [],
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    durations = core.seq_map(lambda b: (b.t1 - b.t0).total_seconds(), blocks)
    assert len(durations) == 3
    assert durations == [
        86400, [], 86400
    ]

def test_seq_map_when():
    # Test case 1: Mapping blocks that meet the condition
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 4), t1=dt.datetime(2023, 1, 5)),
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    result = core.seq_map_when(lambda b: b.t0.day % 2 == 0, lambda b: (b.t1 - b.t0).total_seconds(), blocks)
    assert result == [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        86400,
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]

    # Test case 2: Mapping blocks that all meet the condition
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    result = core.seq_map_when(lambda b: b.t0.day % 2 == 1, lambda b: (b.t1 - b.t0).total_seconds(), blocks)
    assert result == [86400, 86400, 86400]

    # Test case 2: Mapping blocks that none meet the condition
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    result = core.seq_map_when(lambda b: b.t0.day % 2 == 0, lambda b: (b.t1 - b.t0).total_seconds(), blocks)
    assert result == blocks

    # Test case 3: Mapping with nested blocks and None values: should preserve nesting and Nones
    nested_blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 4)),
        [
            core.Block(t0=dt.datetime(2023, 1, 4), t1=dt.datetime(2023, 1, 5)),
            core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
            [],
        ],
        core.Block(t0=dt.datetime(2023, 1, 8), t1=dt.datetime(2023, 1, 9)),
        None,
    ]
    result = core.seq_map_when(
        lambda b: b.t0.day % 2 == 0,
        lambda b: (b.t1 - b.t0).total_seconds(),
        nested_blocks
    )
    assert result == [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 4)),
        [86400, core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)), []],
        86400,
        None,
    ]

def test_seq_trim():
    # case 1
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    t0 = dt.datetime(2023, 1, 2)
    t1 = dt.datetime(2023, 1, 5)
    trimmed_blocks = core.seq_trim(blocks, t0, t1)
    assert trimmed_blocks == [
        None,
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        None
    ]

    # case 2
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)), None],
        [None],
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6)),
    ]
    t0 = dt.datetime(2023, 1, 1, 12, 0, 0)
    t1 = dt.datetime(2023, 1, 5, 12, 0, 0)

    trimmed_blocks = core.seq_trim(blocks, t0, t1)
    assert trimmed_blocks == [
        core.Block(t0=dt.datetime(2023, 1, 1, 12, 0, 0), t1=dt.datetime(2023, 1, 2)),
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)), None],
        [None],
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 5, 12, 0, 0)),
    ]
    assert core.seq_flatten(trimmed_blocks) == [
        core.Block(t0=dt.datetime(2023, 1, 1, 12, 0, 0), t1=dt.datetime(2023, 1, 2)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 5, 12, 0, 0)),
    ]

    t0 = dt.datetime(2023, 1, 10, 12, 0, 0)
    t1 = dt.datetime(2023, 1, 15, 12, 0, 0)

    trimmed_blocks = core.seq_trim(blocks, t0, t1)
    assert trimmed_blocks == [None, [None, None], [None], None]
    assert core.seq_flatten(trimmed_blocks) == []

def test_seq_assert_same_structure():
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        None,
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
         None, core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6))],
        core.Block(t0=dt.datetime(2023, 1, 6), t1=dt.datetime(2023, 1, 7)),
    ]
    filtered_blocks = core.seq_filter(lambda b: b.t0 < dt.datetime(2023, 1, 4), blocks)
    assert len(filtered_blocks) == 4
    core.seq_assert_same_structure(filtered_blocks, blocks)
    with pytest.raises(AssertionError):
        core.seq_assert_same_structure(filtered_blocks, core.seq_flatten(blocks))

def test_seq_replace_block():
    blocks = [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        None,
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
         None, core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6))],
        core.Block(t0=dt.datetime(2023, 1, 6), t1=dt.datetime(2023, 1, 7)),
    ]
    new_blocks = core.seq_replace_block(
        blocks, core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 4)),
        core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 5))
    )
    assert new_blocks == [
        core.Block(t0=dt.datetime(2023, 1, 1), t1=dt.datetime(2023, 1, 2)),
        None,
        [core.Block(t0=dt.datetime(2023, 1, 3), t1=dt.datetime(2023, 1, 5)),
         None, core.Block(t0=dt.datetime(2023, 1, 5), t1=dt.datetime(2023, 1, 6))],
        core.Block(t0=dt.datetime(2023, 1, 6), t1=dt.datetime(2023, 1, 7)),
    ]
