from Directories import network_dir, home
from DataFrame.import_excel_dfs import df_ant_excluded, df_human
from trajectory_inheritance.get import get

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import os
from itertools import groupby
import operator
from tqdm import tqdm
from copy import copy
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

states = ['ab', 'ac', 'b', 'be1', 'be2', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']


def rotational_distance(angle) -> float:
    angle = np.unwrap(angle)
    total_rotation = sum(abs(angle[1::1] - angle[0:-1:1]))
    return total_rotation


def translational_distance(position) -> float:
    if len(position) == 1 or len(position) == 0:
        return 0
    total_translation = sum(sum(abs(position[1::1] - position[0:-1:1])))
    return total_translation


class PathLength_per_state:
    def __init__(self, x):
        self.x = copy(x)
        self.ts = self.extend_time_series_to_match_frames(time_series_dict[self.x.filename], len(self.x.frames))

    def calculate_path_lengths_per_state(self, state):
        """
        Path length without counting any of the path length within the self.to_exclude states.
        """

        # find indices in which self.ts is state
        indices = [i for i, s in enumerate(self.ts) if s == state]

        # find the ranges of indices that are not equal to state
        ranges = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(operator.itemgetter(1), g))
            ranges.append((group[0], group[-1]))

        # cut out ranges from self.x
        translations = {}
        rotations = {}

        for r in ranges:
            x_new = self.x.cut_off(frame_indices=(r[0], r[1]))
            translations[str(r[0]) + '_' + str(r[1])] = translational_distance(x_new.position)
            rotations[str(r[0]) + '_' + str(r[1])] = rotational_distance(x_new.angle)
        return translations, rotations

    @staticmethod
    def extend_time_series_to_match_frames(ts, frame_len):
        indices_to_ts_to_frames = \
            np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10) for _ in range(frame_len)]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended

    # def smooth(self, sigma=9):
    #
    #     if self.x.solver == 'gillespie':
    #         # raise TypeError('you should not smooth gillespie trajectories!')
    #         print('you should not smooth gillespie trajectories!')
    #     self.x.position[:, 0] = self.smooth_array(self.x.position[:, 0], sigma)
    #     self.x.position[:, 1] = self.smooth_array(self.x.position[:, 1], sigma)
    #     # unwrapped_angle = ConnectAngle(self.angle, self.shape)
    #     unwrapped_angle = np.unwrap(self.x.angle)
    #     self.x.angle = self.smooth_array(unwrapped_angle, sigma)

    def smooth(self, sec_smooth=0.2):

        if self.x.solver == 'gillespie':
            # raise TypeError('you should not smooth gillespie trajectories!')
            print('you should not smooth gillespie trajectories!')
        self.x.position[:, 0] = self.smooth_array(self.x.position[:, 0], sec_smooth * self.x.fps)
        self.x.position[:, 1] = self.smooth_array(self.x.position[:, 1], sec_smooth * self.x.fps)
        # unwrapped_angle = ConnectAngle(self.angle, self.shape)
        unwrapped_angle = np.unwrap(self.x.angle)
        self.x.angle = self.smooth_array(unwrapped_angle, sec_smooth * self.x.fps)

    @staticmethod
    def smooth_array(array, kernel_size=None):
        # make sure the kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        new_array = medfilt(array, kernel_size=int(kernel_size))
        new_array = gaussian_filter(new_array, sigma=kernel_size // 5)
        return new_array

    def test_trajectory(self):
        # calculate total translational and rotational distance for unsmoothed trajectory
        unsmoothed_translational = translational_distance(self.x.position)
        unsmoothed_rotational = rotational_distance(self.x.angle)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(np.unwrap(self.x.angle % (2 * np.pi)))
        axs[1].plot(self.x.position)
        # self.x.play(videowriter=True)

        # calculate total translational and rotational distance for smoothed trajectory
        self.smooth(sec_smooth=2)
        smoothed_translational = translational_distance(self.x.position)
        smoothed_rotational = rotational_distance(self.x.angle)
        axs[0].plot(self.x.angle)
        axs[1].plot(self.x.position)
        # self.x.play(videowriter=True)
        # write smoothed_rotational and unsmoothed_rotational in plt.gcf()
        plt.title(self.x.filename)
        axs[0].set_xlabel('frame')
        axs[0].set_ylabel('angle')
        axs[1].set_xlabel('frame')
        axs[1].set_ylabel('position')
        txt0 = 'smoothed_rotational: ' + str(int(smoothed_rotational)) + '\n' + 'unsmoothed_rotational: ' + str(int(unsmoothed_rotational))
        txt1 = 'smoothed_translation: ' + str(int(smoothed_translational)) + '\n' + 'unsmoothed_translation: ' + str(int(unsmoothed_translational))
        # set txt in bottom right corner of axs[0]
        axs[0].text(0.95, 0.05, txt0, horizontalalignment='right', verticalalignment='bottom', transform=axs[0].transAxes)
        axs[1].text(0.95, 0.05, txt1, horizontalalignment='right', verticalalignment='bottom', transform=axs[1].transAxes)
        # plt.show()

        # calculate_path_lengths_per_state for trajectory
        path_lengths = {}
        for state in states:
            path_lengths[state] = self.calculate_path_lengths_per_state(state)

        # sum up all the path lengths
        total_translational = sum([sum(path_lengths[state][0].values()) for state in states])
        total_rotational = sum([sum(path_lengths[state][1].values()) for state in states])

        # compare the two
        print('unsmoothed translational distance: ', unsmoothed_translational)
        print('smoothed translational distance: ', smoothed_translational)
        print('unsmoothed rotational distance: ', unsmoothed_rotational)
        print('smoothed rotational distance: ', smoothed_rotational)
        print('total translational distance summed: ', total_translational)
        print('total rotational distance summed: ', total_rotational)
        DEBUG = 1


if __name__ == '__main__':
    # date = '2023_06_27'
    # date = 'SimTrjs_RemoveAntsNearWall=False'

    # ######################## TIME SERIES #########################################
    time_series_dict = {}
    with open(home + '\\ConfigSpace\\time_series_ant.json', 'r') as json_file:
        time_series_ant = json.load(json_file)
        json_file.close()

    with open(home + '\\ConfigSpace\\time_series_human.json', 'r') as json_file:
        time_series_human = json.load(json_file)
        json_file.close()

    with open(home + '\\ConfigSpace\\time_series_SimTrjs_RemoveAntsNearWall=False_sim.json', 'r') as json_file:
        time_series_sim0 = json.load(json_file)
        json_file.close()

    with open(home + '\\ConfigSpace\\time_series_SimTrjs_RemoveAntsNearWall=True_sim.json', 'r') as json_file:
        time_series_sim1 = json.load(json_file)
        json_file.close()

    time_series_dict.update(time_series_ant)
    time_series_dict.update(time_series_human)
    time_series_dict.update(time_series_sim0)
    time_series_dict.update(time_series_sim1)

    # # ######################## TRAJECTORY TEST #########################################
    # traj = get('sim_XL_2023-07-09_01-52-02.101New')
    # traj.play()
    # DEBUG = 1
    # # traj = get(df_ant_excluded.iloc[0]['filename'])
    # # traj = get(df_human.iloc[0]['filename'])
    # traj = get('S_SPT_5180005_SSpecialT_1_ants (part 1)')
    # pps = PathLength_per_state(traj)
    # pps.test_trajectory()
    # DEBUG = 1

    # ######################### HUMAN EXPERIMENT #########################################

    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\human_pathlengths_all_states.json'
    path_lengths = {}

    for index, row in tqdm(list(df_human.iterrows())):
        traj = get(row['filename'])
        # filename = 'large_20230219122939_20230219123549'
        filename = 'large_20211006172334_20211006173302'

        traj = get(filename)
        path_lengths[traj.filename] = {}
        print(traj.filename)
        pps = PathLength_per_state(traj)
        pps.smooth(sec_smooth=2)
        for state in tqdm(states, desc=traj.filename):
            path_lengths[traj.filename][state] = pps.calculate_path_lengths_per_state(state)
            # this is (translational, rotational)
        DEBUG = 1

    with open(direct, 'w') as json_file:
        json.dump(path_lengths, json_file)
        json_file.close()

    # ######################### ANTS EXPERIMENT #########################################
    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\ant_pathlengths_all_states.json'
    path_lengths = {}

    for index, row in tqdm(list(df_ant_excluded.iterrows())):
        traj = get(row['filename'])
        path_lengths[traj.filename] = {}
        print(traj.filename)
        pps = PathLength_per_state(traj)
        pps.smooth(sec_smooth=1)
        for state in tqdm(states, desc=traj.filename):
            path_lengths[traj.filename][state] = pps.calculate_path_lengths_per_state(state)
            # this is (translational, rotational)

        DEBUG = 1

    with open(direct, 'w') as json_file:
        json.dump(path_lengths, json_file)
        json_file.close()

    # ######################### ANT SIMULATION #########################################
    date = 'SimTrjs_RemoveAntsNearWall=True'
    df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\' + \
             date + '_sim_pathlengths_all_states.json'
    path_lengths = {}

    for index, row in tqdm(list(df_gillespie.iterrows())):
        traj = get(row['filename'])
        path_lengths[traj.filename] = {}
        print(traj.filename)
        pps = PathLength_per_state(traj)
        for state in tqdm(states, desc=traj.filename):
            path_lengths[traj.filename][state] = pps.calculate_path_lengths_per_state(state)
            # this is (translational, rotational)

        DEBUG = 1

    with open(direct, 'w') as json_file:
        json.dump(path_lengths, json_file)
        json_file.close()

    # ######################### ANT SIMULATION #########################################
    date = 'SimTrjs_RemoveAntsNearWall=False'
    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\' + \
             date + '_sim_pathlengths_all_states.json'
    df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
    path_lengths = {}

    for index, row in tqdm(list(df_gillespie.iterrows())):
        traj = get(row['filename'])
        path_lengths[traj.filename] = {}
        print(traj.filename)
        pps = PathLength_per_state(traj)
        for state in tqdm(states, desc=traj.filename):
            path_lengths[traj.filename][state] = pps.calculate_path_lengths_per_state(state)
            # this is (translational, rotational)
        DEBUG = 1

    with open(direct, 'w') as json_file:
        json.dump(path_lengths, json_file)
        json_file.close()
