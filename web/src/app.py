import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_ace import st_ace

st.set_page_config(layout="wide")

import yaml
import datetime as dt
from schedlib import policies, core
from scheduler_server.configs import get_default_config
from scheduler_server.utils import nested_update
import utils


# =================
# streamlit web
# =================

for k in ['data_orig', 'groups_orig', 'data_trans', 'groups_trans', 'data_merge', 'groups_merge']:
    if k not in st.session_state:
        st.session_state[k] = []

if 'user_config' not in st.session_state:
    st.session_state.user_config = {}

def on_load_schedule():
    t0 = dt.datetime.combine(start_date, start_time).astimezone(dt.timezone.utc)
    t1 = dt.datetime.combine(end_date, end_time).astimezone(dt.timezone.utc)

    seqs = policy.init_seqs(t0, t1)
    data, groups = utils.seq2visdata_flat(seqs)
    st.session_state.data_orig = data
    st.session_state.groups_orig = groups

    seqs = policy.transform(seqs)
    data, groups = utils.seq2visdata_flat(seqs)
    st.session_state.data_trans = data
    st.session_state.groups_trans = groups

    seqs = policy.merge(seqs)
    data = core.seq_map(utils.block2dict, seqs)
    st.session_state.data_merge = data

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

config = get_default_config('basic')
config = nested_update(config, st.session_state.user_config)
policy = policies.BasicPolicy(**config)
on_load_schedule()

st.markdown("## Initial Sequences")
timeline_orig = st_timeline(st.session_state.data_orig, st.session_state.groups_orig, key='orig')
st.markdown("## Transformed Sequences")
timeline_trans = st_timeline(st.session_state.data_trans, st.session_state.groups_trans, key='trans')
st.markdown("## Output Schedule")
timeline_merge = st_timeline(st.session_state.data_merge, key='merge')
st.markdown("## Configuration")
config = yaml.safe_load(st_ace(yaml.dump(config), language='yaml', key='config'))
# put it in the session so next reload will use this config
st.session_state.user_config = config