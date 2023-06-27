import numpy as np
import json
from Setup.Maze import Maze
import pandas as pd
from tqdm import tqdm
from DataFrame.import_excel_dfs import df_ant
from Directories import network_dir, df_minimal_dir
from os import path
from matplotlib import pyplot as plt
from copy import copy
from Analysis.Efficiency.PathLength import PathLength
from pandas import DataFrame
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'h']

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

dt = 0.25

df_ant['size'][(df_ant['average Carrier Number'] == 1) & (df_ant['size'] == 'S')] = 'Single (1)'
df_ant['size'][(df_ant['average Carrier Number'] > 1) & (df_ant['size'] == 'S')] = 'S (> 1)'

first_passage_df = pd.DataFrame(columns=['filename', 'size', 'total'] + states)
first_passage_df['filename'] = df_ant['filename']
first_passage_df['size'] = df_ant['size']

minimal_pL = {'XL': 64.57, 'L': 32.94, 'M': 16.48, 'S (> 1)': 8.66, 'Single (1)': 8.66, 'S': 8.66}


def time_first_passage() -> pd.DataFrame:
    # for every filename, find the time_series
    for filename in df_ant['filename']:
        time_series = time_series_dict[filename]
        # for every state, find the first passage time
        for state in states:
            if state not in time_series:
                first_passage_df.loc[first_passage_df['filename'] == filename, state] = \
                df_ant[df_ant['filename'] == filename]['time [s]'] / 60
            else:
                # find the first time the state is reached
                first_passage_df.loc[first_passage_df['filename'] == filename, state] = \
                    np.where(np.array(time_series) == state)[0][0] * dt / 60
    return first_passage_df


def pL_first_passage() -> pd.DataFrame:
    for filename in tqdm(df_ant['filename']):
        ts = time_series_dict[filename]
        x = get(filename)
        ts_new = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)

        kernel_size = 2 * (x.fps // 4) + 1
        x.position, x.angle = x.smoothed_pos_angle(x.position, x.angle, kernel_size=kernel_size)
        total = PathLength(x).total_dist() / minimal_pL[x.size]
        first_passage_df.loc[first_passage_df['filename'] == filename, 'total'] = total

        for state in states[1:]:
            traj = copy(x)
            if state in ts_new:
                frames = [0, np.where(np.array(ts_new) == state)[0][0]]
                traj = traj.cut_off(frame_indices=frames)
                # print(state)
                # traj.play()
                first_passage_df.loc[first_passage_df['filename'] == filename, state] = \
                    PathLength(traj).total_dist(smooth=False) / minimal_pL[traj.size]

        DEBUG = 1
    return first_passage_df


if __name__ == '__main__':
    # first_passage_df = time_first_passage()
    # first_passage_df = pL_first_passage()
    # first_passage_df.to_excel('pL_first_passage_df.xlsx')
    # load the dataframe
    first_passage_df = pd.read_excel('pL_first_passage_df.xlsx')

    # 'Single (1)'
    for state in tqdm(states[1:]):
        fig, axs = plt.subplots(1, 4, sharey=True, sharex=True)
        for size, ax in zip(['XL', 'L', 'M', 'S (> 1)', ], axs):
            print(state)
            group = first_passage_df[first_passage_df['size'] == size]
            # if size == 'S (> 1)' and state == 'ab':
            #     # exclude 'S_SPT_4740002_SSpecialT_1_ants (part 1)' because it is an outlier
            #     group = group[group['filename'] != 'S_SPT_4740002_SSpecialT_1_ants (part 1)']
            # if there are values that are not nan
            if group[state].notna().any():
                z = ax.hist(group[state], bins=15, label=size, density=True, range=[0, 30])
            # percent which where notnan
            percent = group[state].notna().sum() / len(group[state])
            title = size + ', ' + str(round(percent * 100, 1)) + '%'
            # write title vertically
            # ax.text(0.5, 1.08, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.set_ylim([0, 0.6])
        # save the figure
        axs[2].set_xlabel('pL passed before reaching ' + state + ' [minimal pL]')
        axs[0].set_ylabel('probability density')
        plt.tight_layout()


        plt.savefig('results\\state_passage_statistics\\' + 'pL_before_reaching_state_' + state + '.png')
