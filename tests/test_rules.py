import datetime as dt
from schedlib import rules, source as src, core, utils, instrument as inst
import pytest

def test_altrange():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    rule = rules.AltRange(
        alt_range=(45, 60) 
    )
    blocks = src.source_get_blocks('uranus', t0, t1)
    blocks = rule.apply(blocks)

def test_azrange():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    block = inst.ScanBlock(name='test', t0=t0, t1=t1, alt=50, az=-150, throw=100)
    rule = rules.AzRange(az_range=(0, 360))
    block_res = rule([block])[0]
    assert block_res == inst.ScanBlock(name='test', t0=t0, t1=t1, alt=50, az=210, throw=100)

    block = inst.ScanBlock(name='test', t0=t0, t1=t1, alt=50, az=-50, throw=100)
    rule = rules.AzRange(az_range=(0, 360), trim=True)
    block_res = rule([block])[0]
    assert block_res == inst.ScanBlock(name='test', t0=t0, t1=t1, alt=50, az=0, throw=50)

    rule = rules.AzRange(az_range=(0, 360), trim=False)
    block_res = rule([block])[0]
    assert block_res == None

def test_day_mod():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    rule = rules.DayMod(
        day=1,
        day_mod=2,
        day_ref=dt.datetime(2014, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    )
    blocks = src.source_get_blocks('uranus', t0, t1)
    assert len(blocks) == 4
    blocks = rule.apply(blocks)
    assert len(core.seq_flatten(blocks)) == 2

def test_drift_mode():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    rule = rules.DriftMode(
       mode='rising'
    )
    blocks = src.source_get_blocks('uranus', t0, t1)
    assert len(blocks) == 4
    blocks = rule.apply(blocks)
    assert len(core.seq_flatten(blocks)) == 2
    for block in core.seq_flatten(blocks):
        assert block.mode == 'rising'

    with pytest.raises(ValueError):
        rule = rules.DriftMode(mode='something-else') 

def test_min_duration():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    blocks = [
        core.Block(t0=t0,t1=t0+dt.timedelta(seconds=5)),
        core.Block(t0=t0,t1=t0+dt.timedelta(seconds=15))]
    rule = rules.MinDuration(
        min_duration=10
    )
    blocks = rule.apply(blocks)
    assert blocks == [
        None,
        blocks[1]
    ]

def test_sun_avoidance():
    t0 = dt.datetime(2022, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2022, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    blocks = src.source_get_blocks('sun', t0, t1)
    rule = rules.SunAvoidance(
        min_angle_az = 3,
        min_angle_alt = 3,
        n_buffer = 0,
        time_step = 30, 
    )
    assert len(blocks) == 6
    assert rule.apply(blocks) == [None] * len(blocks)

def test_make_source():
    t0 = dt.datetime(2022, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2022, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    blocks = src.source_gen_seq('jupiter', t0, t1)
    rule = rules.MakeSourcePlan(
        specs = [{'bounds_x': [-0.5, 0.5], 'bounds_y': [-0.5, 0.5]}],
        spec_shape = 'ellipse',
        max_obs_length = 1200, # seconds
    )
    assert len(blocks) == 3
    new_blocks = core.seq_flatten(rule.apply(blocks))
    assert new_blocks != blocks
    rule = rules.MakeSourceScan(
        preferred_length = 1000,
        rng_key = utils.PRNGKey(42)
    )
    new_blocks2 = core.seq_flatten(rule.apply(new_blocks))
    assert new_blocks2 != new_blocks
