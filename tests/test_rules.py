#%%
import datetime as dt
from schedlib import rules, source as src
# from equinox import tree_pprint


def test_altrange():
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    rule = rules.AltRange(
        alt_range=(45, 60) 
    )
    blocks = src.source_get_blocks('uranus', t0, t1)
    blocks = rule.apply(blocks)