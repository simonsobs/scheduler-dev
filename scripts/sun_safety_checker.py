#!/usr/bin/env python

import numpy as np
import argparse
import datetime

## Code to check sun safety

satp1_policy_info = {'az_limits': [-45, 405], 'min_sun_angle': 41, 'min_sun_time': 1980}

parser = argparse.ArgumentParser()
parser.add_argument('sched_path')

class SunCrawler:
    def __init__(self, path):
        self.schedf = open(path, 'r')

        self.cur_time = 0
        self.cur_az = 0
        self.cur_el = 0

        self._get_initial_pos()
        self._generate_sun_solution()
        #self._test_sungod()

    def _move_to_parse(self, l):
        try:
            az = float(l.split('az=')[1].split(',')[0])
        except IndexError:
            print('Bad input!', l)
            az = None

        try:
            el = float(l.split('el=')[1].rstrip(')\n'))
        except IndexError:
            try:
                el = float(l.split(',')[1].rstrip(')\n'))
            except IndexError:
                print('Bad input!', l)
                el = None

        return az, el

    def _wait_parse(self, l):
        t = l.split("('")[1].rstrip("')\n").split(",")[0].strip("'")
        time = datetime.datetime.fromisoformat(t).timestamp()
        return time

    def _scan_parse(self, b, debug=False):
        #print(b)
        planet_flag = np.any([('now = ' in l) for l in b]) or np.any([("'cal'" in l) for l in b])

        start_time = self.cur_time
        stop = 0.
        width = 0.
        drift = 0.
        for l in b:
            if 'stop_time' in l:
                stop = datetime.datetime.fromisoformat(l.split("stop_time='")[1].rstrip("', \n")).timestamp()
            if 'width' in l and 'drift' in l:
                width = float(l.split(', ')[0].split('=')[1])
                drift = float(l.split(', ')[1].split('=')[1].rstrip(', \n'))
            elif 'width' in l and 'drift' not in l:
                width = float(l.split('=')[1].rstrip(', \n'))
            elif 'drift' in l and 'width' not in l:
                drift = float(l.split('=')[1].rstrip(', \n'))
                
        init_az_range = [self.cur_az, self.cur_az + width]
        drifted_az = self.cur_az + drift * (stop - self.cur_time)
        end_az_range = [drifted_az, drifted_az + width]

        d1 = self.sungod.check_trajectory(init_az_range, [self.cur_el, self.cur_el], t=self.cur_time)
        d2 = self.sungod.check_trajectory(end_az_range, [self.cur_el, self.cur_el], t=stop)
        assert((d1['sun_dist_min'] > self.satp1_policy['exclusion_radius']) and (d2['sun_dist_min'] > self.satp1_policy['exclusion_radius']))

        #print('Min scan distance to sun at start', d1['sun_dist_min'])
        print('Min scan distance to sun at end', d2['sun_dist_min'])

        if debug:
            print('azimuth : ', self.cur_az, ' --> ', drifted_az + width)
            print('timestamp : ', self.cur_time, ' --> ', stop)
        
        self.cur_az = drifted_az + width
        self.cur_time = stop

    def _get_initial_pos(self):
        time_flag = False
        pos_flag = False
        while True:
            l = self.schedf.readline()
            if 'move_to' in l:
                az, el = self._move_to_parse(l)
                self.cur_az = az
                self.cur_el = el
                pos_flag = True
                
            if 'wait_until' in l:
                ts = self._wait_parse(l)
                self.cur_time = ts
                time_flag = True

            if pos_flag and time_flag:
                break

    def _generate_sun_solution(self):
        import sys
        sys.path.append('/so/home/ktcrowley/repos/')
        import socs.socs.agents.acu.avoidance as avoidance

        self.satp1_policy = avoidance.DEFAULT_POLICY
        self.satp1_policy['min_el'] = 48.
        self.satp1_policy['min_sun_time'] = satp1_policy_info['min_sun_time']
        self.satp1_policy['exclusion_radius'] = satp1_policy_info['min_sun_angle']

        self.sungod = avoidance.SunTracker(policy=self.satp1_policy, base_time=self.cur_time, compute=True)

    def _test_sungod(self):
        out = self.sungod.get_sun_pos(t=self.cur_time+500.)
        print(out)

    def _get_traj(self, az, el):
        if az == 0:
            az = self.next_az

        # catching wrap questions
        if self.cur_az < 180 and az >= 180: # ccw wrap
            mid_az = np.mean([self.cur_az, az])
        elif self.cur_az > 180 and az <= 180: # cw wrap
            mid_az = np.mean([self.cur_az, az])
        else:
            mid_az = az

        mid_el = np.mean([self.cur_el, el])

        return [self.cur_az, mid_az, az], [self.cur_el, mid_el, el]

    def step_thru_schedule(self, debug=False):
        cur_block = []
        scan_flag = False
        
        while True:
            l = self.schedf.readline()

            if 'wait_until' in l:
                ts = self._wait_parse(l)
                if debug:
                    print('timestamp: ', self.cur_time, ' --> ', ts)
                self.cur_time = ts
            
            if 'move_to' in l:
                az, el = self._move_to_parse(l)

                if az is not None and el is not None:
                    az_range, el_range = self._get_traj(az, el)
                    if debug:
                        print('azimuth :', self.cur_az, ' --> ', az)
                        print('elevation :', self.cur_el, ' --> ', el)
                    
                    self.cur_az = az
                    self.cur_el = el
                elif el is not None:
                    az_range, el_range = self._get_traj(0, el)

                    if debug:
                        print('azimuth :', self.cur_az, ' --> ', self.next_az)
                        print('elevation :', self.cur_el, ' --> ', el)

                    self.cur_az = self.next_az
                    self.cur_el = el
                    
                #print(az_range, el_range)
                d = self.sungod.check_trajectory(az_range, el_range, t=self.cur_time)
                print('Min slew distance to sun', d['sun_dist_min'])
                assert(d['sun_dist_min'] > self.satp1_policy['exclusion_radius'])

            if 'az = ' in l:
                az = float(l.split('az = ')[1].split('+')[0])
                try:
                    self.next_az = az
                except AttributeError:
                    self.next_az = 0.
                    self.next_az = az

            if 'run.seq.scan' in l:
                assert(l[-2:] == '(\n')
                scan_flag = True

            if l.strip() == ')':
                scan_flag = False
                self._scan_parse(cur_block, debug=debug)
                cur_block = []

            if scan_flag:
                cur_block.append(l)

            if l == '':
                break
                

if __name__ == "__main__":
    args = parser.parse_args()
    sc = SunCrawler(args.sched_path)
    sc.step_thru_schedule(debug=True)
