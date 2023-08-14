from Directories import network_dir, home
from DataFrame.import_excel_dfs import df_ant_excluded, df_human
from trajectory_inheritance.get import get

import os
from itertools import groupby
from tqdm import tqdm
from copy import copy
import json
import numpy as np
import pandas as pd

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']


class BackRoomFlipping:
    def __init__(self, x):
        self.x = copy(x)
        self.filename = self.x.filename
        self.ts = self.extend_time_series_to_match_frames(time_series_dict[self.x.filename], len(self.x.frames))
        self.ss = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.ts])]
        self.decisions = self.find_backroom_flipping()

    @staticmethod
    def extend_time_series_to_match_frames(ts, frame_len):
        indices_to_ts_to_frames = \
            np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10) for _ in range(frame_len)]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended

    def time_stamped_series(self, ts) -> list:
        groups = groupby(ts)
        return [(label, sum(1 for _ in group) * 1 / self.x.fps) for label, group in groups]

    def find_backroom_flipping(self):
        # reduce all 'ab' and 'ac' to 'a'
        ts = self.ts
        ts = ['a' if s in ['ab', 'ac'] else s for s in ts]
        ts = ['b' if s in ['b1', 'b2', 'be'] else s for s in ts]
        ts = ['c' if s in ['cg', 'c'] else s for s in ts]
        tss_backroom = self.time_stamped_series(ts)

        # find all the elements following an 'a' element
        after_a = []
        for i, s in enumerate(tss_backroom):
            if s[0] == 'a':
                if i < len(tss_backroom) - 1:
                    after_a.append(tss_backroom[i])
                    after_a.append(tss_backroom[i + 1])
        return after_a

    @staticmethod
    def reduce_decisions(decisions, min_time=20):
        decisions = [decisions[0]] + [d if d[1] > min_time else (None, d[1]) for d in decisions[1:]]

        decisions_reduced = [decisions[0]]
        # add up the second element all successive pairs with the same first element
        for d2 in decisions[1:]:
            d1 = decisions_reduced[-1]
            if d2[0] is None:
                decisions_reduced[-1] = (d1[0], d1[1] + d2[1])
            elif d1[0] == d2[0]:
                decisions_reduced[-1] = (d1[0], d1[1] + d2[1])
            else:
                decisions_reduced.append(d2)
        return decisions_reduced

    @staticmethod
    def percentage_backroom_flipped(decisions):  # remove all the 'a' decisions
        dec = [d for d in decisions if d[0] != 'a']
        flipped = [i for i, s in enumerate(dec) if i < len(dec) - 1 and s[0] != dec[i + 1][0]]
        if len(dec) < 2:
            return None
        return len(flipped) / (len(dec) - 1)

    @staticmethod
    def total_entrances(decisions):
        dec = [d for d in decisions if d[0] != 'a']
        return len(dec)


if __name__ == '__main__':

    # date = '2023_06_27'
    # date = 'SimTrjs_RemoveAntsNearWall=False'
    date = 'SimTrjs_RemoveAntsNearWall=True'
    df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')

    ts_directory = home + '\\Gillespie\\' + date + '_sim_time_series.json'
    if not os.path.exists(os.path.join(home, ts_directory)):
        raise TypeError(r'Run C:\Users\tabea\PycharmProjects\AntsShapes\DataFrame\gillespie_dataFrame.py')

    with open(ts_directory, 'r') as json_file:
        time_series_sim_dict = json.load(json_file)
        json_file.close()

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    time_series_dict.update(time_series_sim_dict)

    # # ######################## TRAJECTORY TEST #########################################
    # # traj = get('sim_XL_2023-07-09_01-52-02.101New')
    # traj = get('XL_SPT_5040013_XLSpecialT_1_ants')
    # # # traj = get(df_human.iloc[0]['filename'])
    # # traj.play(step=200)  # videowriter=True
    # brf = BackRoomFlipping(traj)
    # rd = brf.reduce_decisions(brf.decisions, min_time=20)
    # brf.percentage_backroom_flipped(rd)
    # DEBUG = 1

    # ######################### ANT SIMULATION #########################################
    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\back_room_flipping\\' + \
             date + '_sim_flipping.json'
    decisions = {}

    for index, row in tqdm(list(df_gillespie.iterrows())):
        traj = get(row['filename'])
        print(traj.filename)
        brf = BackRoomFlipping(traj)
        decisions[traj.filename] = brf.decisions

    # write to json
    with open(direct, 'w') as json_file:
        json.dump(decisions, json_file)
        json_file.close()

    # # # # ######################### ANTS EXPERIMENT #########################################
    # direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\back_room_flipping\\solver_flipping.json'
    # decisions = {}
    #
    # for index, row in tqdm(list(df_ant_excluded.iterrows())):
    #     traj = get(row['filename'])
    #     print(traj.filename)
    #     brf = BackRoomFlipping(traj)
    #     decisions[traj.filename] = brf.decisions
    #
    # # write to json
    # with open(direct, 'w') as json_file:
    #     json.dump(decisions, json_file)
    #     json_file.close()
