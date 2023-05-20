import streamlit as st
from streamlit_timeline import st_timeline
st.set_page_config(layout="wide")

from schedlib import policies, core
import utils

import os.path as op
import datetime as dt

minute = 60
hour = 60 * minute

config = {
    'master_schedule': op.dirname(__file__) + '/schedule_sat.txt',
    'rules': {
        'sun-avoidance': {
            'min_angle_az': 6,  # deg
            'min_angle_alt': 6, # deg
            'time_step': 10,    # sec
            'n_buffer': 3,     # 30 * 1 = 0.5 mins
        },
        'day-mod': {
            'day': 0,
            'day_mod': 1,
            'day_ref': dt.datetime(2014, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        },
        'calibration-min-duration': {
            'min_duration': 5 * minute,
        },
        'make-source-plan': {
            'specs': [{'bounds_x': [-0.5, 0.5], 'bounds_y': [-0.5, 0.5]}],  # test
            'spec_shape': 'ellipse',
            'max_obs_length': 6000,
        },
        'make-source-scan': {
            'preferred_length': 1800,
        }
    },
    'calibration_targets': ["uranus", "saturn"],
    'soft_targets': []
}
policy = policies.BasicPolicy(**config)

# =================
# streamlit web
# =================

if "data" not in st.session_state:
    st.session_state.data = []

if "groups" not in st.session_state:
    st.session_state.groups = []

def on_load_schedule():
    t0 = dt.datetime.combine(start_date, start_time).astimezone(dt.timezone.utc)
    t1 = dt.datetime.combine(end_date, end_time).astimezone(dt.timezone.utc)
    seqs = policy.init_seqs(t0, t1)
    seqs = policy.transform(seqs)
    data, groups = utils.seq2visdata_flat(seqs)
    st.session_state.data = data
    st.session_state.groups = groups

st.title("SO Scheduler Web")

with st.sidebar:
    st.subheader("Schedule")
    now = dt.datetime.utcnow()

    start_date = now.date()
    start_time = now.time()
    end_date = start_date + dt.timedelta(days=1)
    end_time = start_time
    start_date = st.date_input("Start date", value=start_date)
    start_time = st.time_input("Start time (UTC)", value=start_time)
    end_date = st.date_input("End date", value=end_date)
    end_time = st.time_input("End time (UTC)", value=end_time)

    st.button("Load Schedule", on_click=on_load_schedule)

timeline = st_timeline(st.session_state.data, st.session_state.groups)
