import pytest

from schedlib import source as src
import ephem
import datetime as dt
import numpy as np

def test_Location():
    # Test case 1: Verify at() method returns a new ephem.Observer object with correct attributes
    location = src.Location(lat=51.5074, lon=-0.1278, elev=0)
    date = dt.datetime(2023, 1, 1)
    observer = location.at(date)
    assert isinstance(observer, ephem.Observer)
    assert observer.date == ephem.date(date)

    # Test case 2: Verify at() method returns a new ephem.Observer object for a different date
    location = src.Location(lat=40.7128, lon=-74.0060, elev=0)
    date1 = dt.datetime(2023, 1, 1)
    observer1 = location.at(date1)
    date2 = dt.datetime(2023, 1, 2)
    observer2 = location.at(date2)
    assert isinstance(observer2, ephem.Observer)
    assert observer2.date == ephem.date(date2)
    assert observer1 is not observer2

def test_get_source():
    source = src.get_source('sun')
    assert isinstance(source, ephem.Body)

def test_source_get_az_alt():
    # Test case 1: Verify azimuth and altitude values for a single time
    source = 'sun'
    times = [dt.datetime(2023, 1, 1)]
    expected_az = [4.19157791]
    expected_alt = [-0.15743408]
    az, alt = src._source_get_az_alt(source, times)
    assert np.allclose(az, expected_az)
    assert np.allclose(alt, expected_alt)

    # Test case 2: Verify azimuth and altitude values for multiple times
    source = 'jupiter'
    times = [dt.datetime(2023, 1, 1, 0, 0, 0), dt.datetime(2023, 1, 2, 0, 0, 0), dt.datetime(2023, 1, 3, 0, 0, 0)]
    expected_az = [5.27258301, 5.25825977, 5.24444818]
    expected_alt = [0.9176504,  0.90511781, 0.89251304]
    az, alt = src._source_get_az_alt(source, times)
    assert np.allclose(az, expected_az)
    assert np.allclose(alt, expected_alt)

def test_source_get_blocks():
    source = 'sun'
    t0 = dt.datetime(2023, 1, 1, 0, 0, 0)
    t1 = dt.datetime(2023, 1, 1, 12, 0, 0)
    blocks = src.source_get_blocks(source, t0, t1)
    assert blocks == [
        src.SourceBlock(
            t0=dt.datetime(2022, 12, 31, 9, 48, 9, 902594),
            t1=dt.datetime(2022, 12, 31, 16, 34, 11, 308130),
            name='sun',
            mode='rising'
        ),
        src.SourceBlock(
            t0=dt.datetime(2022, 12, 31, 16, 34, 11, 308130),
            t1=dt.datetime(2022, 12, 31, 23, 20, 7, 342349),
            name='sun',
            mode='setting'
        ),
        src.SourceBlock(
            t0=dt.datetime(2023, 1, 1, 9, 48, 48, 46182),
            t1=dt.datetime(2023, 1, 1, 16, 34, 39, 671616),
            name='sun',
            mode='rising'
        ),
        src.SourceBlock(
            t0=dt.datetime(2023, 1, 1, 16, 34, 39, 671616),
            t1=dt.datetime(2023, 1, 1, 23, 20, 25, 379070),
            name='sun',
            mode='setting'
        )
    ]

def test_precomputed_source():
    t0 = dt.datetime(2023, 1, 1, 0, 0, 0)
    t1 = dt.datetime(2023, 1, 1, 12, 0, 0)
    source = src.PrecomputedSource.for_('uranus', t0=t0, t1=t1, buf=dt.timedelta(days=0))
    assert isinstance(source, src.PrecomputedSource)
    assert len(source.blocks) == 4

    t = int(dt.datetime(2023, 1, 1, 5, 0, 0).timestamp())
    assert np.allclose(source.interp_az(t), [0.27020165])
    assert np.allclose(source.interp_alt(t), [5.15476608])

    assert 'uranus' in src.PRECOMPUTED_SOURCES

def test_source_block():
    srcblk = src.SourceBlock(
        t0=dt.datetime(2023, 1, 1, 0, 0, 0),
        t1=dt.datetime(2023, 1, 1, 12, 0, 0),
        name='sun',
        mode='rising'
    )
    assert isinstance(srcblk, src.SourceBlock)

    with pytest.raises(ValueError):
        srcblk = src.SourceBlock(
            t0=dt.datetime(2023, 1, 1, 0, 0, 0),
            t1=dt.datetime(2023, 1, 1, 12, 0, 0),
            name='sun',
            mode='rise'
        )

def test_source_gen_seq():
    blocks = src.source_gen_seq('uranus', dt.datetime(2023, 1, 1, 0, 0, 0), dt.datetime(2023, 1, 1, 12, 0, 0))
    assert blocks == [
        src.SourceBlock(
            t0=dt.datetime(2023, 1, 1, 0, 0),
            t1=dt.datetime(2023, 1, 1, 0, 40, 38, 505631),
            name='uranus',
            mode='rising'
        ),
        src.SourceBlock(
            t0=dt.datetime(2023, 1, 1, 0, 40, 38, 505631),
            t1=dt.datetime(2023, 1, 1, 6, 14, 17, 34179),
            name='uranus',
            mode='setting'
        )
    ]

def test_source_block_get_az_alt():
    srcblk = src.SourceBlock(
        t0=dt.datetime(2023, 1, 1, 0, 40, 38),
        t1=dt.datetime(2023, 1, 1, 1, 14, 17),
        name='uranus',
        mode='setting'        
    )
    times, az, alt = src.source_block_get_az_alt(srcblk)
    assert len(times) == 2019
    assert np.allclose(az[:5], [0.89042902, 0.89042902, 0.89042902, 0.89042896, 0.89042896])
    assert np.allclose(alt[:5], [5.63179383e-05, 6.28313017e+00, 6.28301907e+00, 6.28290749e+00, 6.28279591e+00])