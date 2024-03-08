from schedlib import policies, config as cfg
from schedlib.instrument import ScanBlock
from schedlib import rules, source as src

import argparse
import copy
import os
import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from schedlib.policies import TSATPolicy
from schedlib import utils as u
from schedlib import instrument as inst

def make_config(pos='top', elevation=50, caltype='beam',user_mvs=[]):
    ufm_mv12_shift = np.degrees([0, 0])
    ufm_mv35_shift = np.degrees([0, 0])
    ufm_mv23_shift = np.degrees([0, 0])
    ufm_mv5_shift  = np.degrees([0, 0])
    ufm_mv27_shift = np.degrees([0, 0])
    ufm_mv33_shift = np.degrees([0, 0])
    ufm_mv17_shift = np.degrees([0, 0])

    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    geometries = {
      'ws3': {
        'center': [-d_xi+ufm_mv12_shift[0], d_eta_side+ufm_mv12_shift[1]],
        'radius': 6,
      },
      'ws2': {
        'center': [-d_xi+ufm_mv35_shift[0], -d_eta_side+ufm_mv35_shift[1]],
        'radius': 6,
      },
      'ws4': {
        'center': [0+ufm_mv23_shift[0], d_eta_mid+ufm_mv23_shift[1]],
        'radius': 6,
      },
      'ws0': {
        'center': [0+ufm_mv5_shift[0], 0+ufm_mv5_shift[1]],
        'radius': 6,
      },
      'ws1': {
        'center': [0+ufm_mv27_shift[0], -d_eta_mid+ufm_mv27_shift[1]],
        'radius': 6,
      },
      'ws5': {
        'center': [d_xi+ufm_mv33_shift[0], d_eta_side+ufm_mv33_shift[1]],
        'radius': 6,
      },
      'ws6': {
        'center': [d_xi+ufm_mv17_shift[0], -d_eta_side+ufm_mv17_shift[1]],
        'radius': 6,
      },
    }

    # top, bottom means the position on the sky
    left_boresight_0 = 'ws3,ws2'
    middle_boresight_0 = 'ws0,ws1,ws4'
    right_boresight_0 = 'ws5,ws6'
    bottom_boresight_0 = 'ws1,ws2,ws6'
    top_boresight_0 = 'ws3,ws4,ws5'

    mv2ws = {'5':'0',
             '27':'1',
             '35':'2',
             '12':'3',
             '23':'4',
             '33':'5',
             '17':'6',
             }
             
    ufms = {
        'left': ['ws3,ws2', 'Mv12Mv35'],
        'right': ['ws5,ws6', 'Mv17Mv33'],
        'middle': ['ws0,ws1,ws4', 'Mv27Mv5Mv23'],
        'bottom': ['ws1,ws2,ws6', 'Mv17Mv27Mv35'],
        'top': ['ws3,ws4,ws5', 'Mv33Mv23Mv12'],
        'center': ['ws0', 'Mv5'],
        'bottombottom':['ws1','Mv27'],
        'mv27mv5':['ws0,ws1','Mv5Mv27']
    }

    if user_mvs:
        ufms = {}
        wsstr=''
        mvstr=''
        for mv in user_mvs:
            wsstr+='ws%s,'%(mv2ws[mv])
            mvstr+='Mv%s,'%(mv)
        wsstr=wsstr[:-1]
        mvstr=mvstr[:-1]
        ufms['userinput']=[wsstr,mvstr]
            
    blocks = {
        'calibration': {
            'saturn': {
                'type' : 'source',
                'name' : 'saturn',
            },
            'jupiter': {
                'type' : 'source',
                'name' : 'jupiter',
            },
            'moon': {
                'type' : 'source',
                'name' : 'moon',
            },
            'taua': {
                'type' : 'source',
                'name' : 'taua',
            },
            'galcenter': {
                'type' : 'source',
                'name' : 'galcenter',
            },
        },
        'baseline': {
            'cmb': {
                'type': 'toast',
                #'file': '/so/home/kmharrin/public_html/observation_scripts/baseline_schedule_20231031.txt'
                'file': '/so/home/ykyohei/public_html/schedule/E60_A40_S2023-12-01_F2025-01-01_D-10_-40_L0.86_6.86_12.86_18.86_T65.00_M045_S045.txt'
            }
        }
    }

    # target, location,elevation, boresight, tag-to-add-to-data-files
    cal_targets = {
        'beam': [
            ('moon', ufms[pos][0], elevation, 0, ufms[pos][1]),
            ('jupiter', ufms[pos][0], elevation, 0, ufms[pos][1]),
            ('saturn', ufms[pos][0], elevation, 0, ufms[pos][1]),
        ],
        'pol': [
            ('taua', ufms[pos][0], elevation, 0, ufms[pos][1]),
            ('galcenter', ufms[pos][0], elevation, 0, ufms[pos][1]),
        ],
        'baseline': [],
    }

    config = {
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'sun-avoidance': {
                'min_angle_az': 49, #keep-out angle, i.e. tells script not to build scans within 49 degrees of the sun
                'min_angle_alt': 49,
            },
            'min-duration': {
                'min_duration': 600
            },
            'az-range': {
                'az_range': [-10, 400], #absolute min and max program can use
                'trim': True #If set to true, if doing a large az scan (large az drift), will stay within the az bounds specified
            },
        },
        'allow_partial':True,
        'cal_targets': cal_targets[caltype],
        'merge_order': {'pol': ['taua', 'galcenter'], 'beam':['moon','jupiter', 'saturn'], 'baseline': 'baseline'}[caltype],
        'time_costs': {
            'det_setup': 40*60,
            'bias_step': 60,
            'ufm_relock': 15*60,
        },
        'ufm_relock': True,
        'scan_tag': None,
    }
    print(cal_targets[caltype])

    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', default=None,help='if a custom date (i.e. not today) is desired. format YYYY-MM-DD')
    parser.add_argument('--output-dir','-o', type=str, default='./', help='output directory')
    parser.add_argument('--elevation',type=float,default=None,help='Run calibration schedules at a specific elevation.')
    parser.add_argument('--mvs',nargs='+',default=[],help='User input Mv list to observe; i.e. 5 27 23 will run center wafers.')
    
    args = parser.parse_args()

    # default is today
    today = dt.date.today()
    tomrw = today + dt.timedelta(days=1)
    t0 = dt.datetime(today.year, today.month, today.day, 6, 0, 0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(tomrw.year, tomrw.month, tomrw.day, 10, 0, 0, tzinfo=dt.timezone.utc)

    if args.day:
        y,m,d = args.day.split('-')
        if m[0]=='0':
            m=m[1]
        if d[0]=='0':
            d=d[1]
        t0 = dt.datetime(int(y), int(m), int(d), 6, 0, 0, tzinfo=dt.timezone.utc)
        nextday=t0+dt.timedelta(days=1)
        t1 = dt.datetime(nextday.year, nextday.month, nextday.day, 10, 0, 0, tzinfo=dt.timezone.utc)
    if not args.elevation:
        cal_el = [40,50,60]
    else:
        cal_el = [args.elevation]

    default_positions = ['left','middle', 'right']
    if args.mvs:
        positions=['userinput']
    else:
        positions=default_positions
    for pos in positions:
        for el in cal_el:
            for caltype in ['beam', 'pol']:
                config = make_config(pos, el, caltype,args.mvs)
                policy = TSATPolicy(**config)

                seqs = policy.init_seqs(t0, t1)
                seqs = policy.apply(seqs)
                #u.pprint(seqs)

                fname = os.path.join(args.output_dir, t0.strftime(f'%Y%m%d_{pos}_el{el:.2f}_{caltype}_observations.py'))
                with open(fname, 'w') as f:
                    f.write(str(policy.seq2cmd(seqs, t0, t1)))
                print(fname +' is written.')

    el, caltype = 60, 'baseline'
    if args.elevation:
        el=args.elevation
    for pos in ['center']:
                config = make_config(pos, el, caltype)
                policy = TSATPolicy(**config)

                seqs = policy.init_seqs(t0, t1)
                seqs = policy.apply(seqs)

                fname = os.path.join(args.output_dir, t0.strftime(f'%Y%m%d_{pos}_el{el:.2f}_{caltype}_observations.py'))
                with open(fname, 'w') as f:
                    f.write(str(policy.seq2cmd(seqs, t0, t1)))
                print(fname +' is written.')
