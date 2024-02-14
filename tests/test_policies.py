from schedlib import policies
import datetime as dt

minute = 60

def test_flex_policy():
    config = """
    blocks:
      master: 
        type: toast
        file: data/schedule_test.txt
      calibration:
        saturn:
          type: source
          name: saturn 
        moon:
          type: source
          name: moon
    rules:
      - name: sun-avoidance
        min_angle: 30
      - name: day-mod
        day: 0
        day_mod: 1
        day_ref: !datetime "2014-01-01 00:00:00"
      - name: make-drift-scan
        block_query: calibration
        array_query: full
        el_bore: 50
        drift: true
    post_rules:
      - name: min-duration
        min_duration: 600
    merge_order:
      - moon
      - saturn
      - master
    geometries:
      full:
        center:
          - 0
          - 0
        radius: 7
    """
    policy = policies.FlexPolicy.from_config(config)
    seqs = policy.init_seqs(
        dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc))
    policy.apply(seqs)
    seqs = policy.transform(seqs)
    seqs = policy.merge(seqs)

    # test drift scan
    seqs = policy.get_drift_scans(
        t0=dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        t1=dt.datetime(2023, 1, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
        el_bore=50,
        array_query='full' 
    )
