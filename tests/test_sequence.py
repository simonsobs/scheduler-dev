import os, os.path as op
from schedlib.parse import parse_sequence_from_toast

def test_parse_sequence_from_toast():
    ifile = op.join(op.abspath(op.dirname(__file__)), "../data/schedule_sat.txt")
    seq = parse_sequence_from_toast(ifile)