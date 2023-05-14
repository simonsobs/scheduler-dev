from schedlib.parse import parse_sequence_from_toast
import os.path as op

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "data/schedule_sat.txt")
    seq = parse_sequence_from_toast(ifile)
    assert len(seq) == 20