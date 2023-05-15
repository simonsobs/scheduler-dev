from scheduler_server import handler
from datetime import datetime, timezone
import os.path as op

def test_dummy():
    t0 = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2023, 1, 2, 0, 0, 10, tzinfo=timezone.utc)
    assert "import time" in handler.dummy_policy(t0, t1, {}) 

def test_basic():
    t0 = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2023, 1, 2, 0, 0, 10, tzinfo=timezone.utc)
    cmds = handler.basic_policy(t0, t1, {
        'master_schedule': op.dirname(__file__) + "/data/schedule_sat.txt"
    })
    assert len(cmds) == 29
