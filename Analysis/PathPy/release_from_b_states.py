import numpy as np
import json
from Setup.Maze import Maze
import pandas as pd
from tqdm import tqdm
from DataFrame.import_excel_dfs import df_ant
from Directories import network_dir, df_minimal_dir, home
from os import path
from matplotlib import pyplot as plt
from copy import copy
from Analysis.Efficiency.PathLength import PathLength
from pandas import DataFrame
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

dt = 0.25

df_ant['size'][(df_ant['average Carrier Number'] == 1) & (df_ant['size'] == 'S')] = 'Single (1)'
df_ant['size'][(df_ant['average Carrier Number'] > 1) & (df_ant['size'] == 'S')] = 'S (> 1)'

first_release_df = pd.DataFrame(columns=['filename', 'size', 'total', 'pL'])
first_release_df['filename'] = df_ant['filename']
first_release_df['size'] = df_ant['size']

minimal_pL = {'XL': 64.57, 'L': 32.94, 'M': 16.48, 'S (> 1)': 8.66, 'Single (1)': 8.66, 'S': 8.66}

to_exclude = ['L_SPT_4420007_LSpecialT_1_ants (part 1)',
              'L_SPT_5030006_LSpecialT_1_ants (part 1)',
              'L_SPT_4420005_LSpecialT_1_ants (part 1)',
              'L_SPT_4670001_LSpecialT_1_ants (part 1)',
              'L_SPT_4420010_LSpecialT_1_ants (part 1)',
              'M_SPT_4700003_MSpecialT_1_ants (part 1)',
              'L_SPT_4670006_LSpecialT_1_ants (part 1)',
              'L_SPT_5030001_LSpecialT_1_ants (part 1)',
              'S_SPT_4760017_SSpecialT_1_ants (part 1)',
              'L_SPT_5010001_LSpecialT_1_ants (part 1)']
# the first three are especially long


def get_indices(time_series) -> list:
    is_in_states = np.isin(time_series, states)
    succession_indices = np.flatnonzero(np.concatenate(([False], np.diff(is_in_states.astype(int)) != 0))).tolist()
    if len(succession_indices) > 0:
        if len(succession_indices) % 2 == 1:
            succession_indices.append(len(time_series))
        # split the succession_indices into pairs as type list
        pairs = [(succession_indices[i], succession_indices[i + 1]) for i in range(0, len(succession_indices), 2)]
    else:
        pairs = []
    return pairs


def time_first_release() -> pd.DataFrame:
    # for every filename, find the time_series
    for filename in df_ant['filename']:
        time_series = time_series_dict[filename]
        # for every state, find the first passage time
        pairs = get_indices(time_series)
        raise NotImplementedError
        first_release_df.loc[first_release_df['filename'] == filename, 'pL'] = \
            (release_index - entrance_index) * dt / 60
    return first_release_df


def pL_first_release() -> pd.DataFrame:
    first_release_dict = {}
    for filename in tqdm(df_ant['filename']):
        ts = time_series_dict[filename]
        x = get(filename)
        time_series = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)

        kernel_size = 2 * (x.fps // 4) + 1
        x.position, x.angle = x.smoothed_pos_angle(x.position, x.angle, kernel_size=kernel_size)
        total = PathLength(x).total_dist() / minimal_pL[x.size]
        first_release_df.loc[first_release_df['filename'] == filename, 'total'] = total
        first_release_dict[filename] = []
        pairs = get_indices(time_series)
        for pair in pairs:
            if np.abs(np.diff(pair)) > 5 * x.fps:
                traj = copy(x)
                traj = traj.cut_off(frame_indices=pair)
                # print(state)
                # traj.play()
                first_release_dict[filename].append(
                    PathLength(traj).total_dist(smooth=False) / minimal_pL[traj.size])
    first_release_df['pL'] = first_release_df['filename'].map(first_release_dict)
    return first_release_df


if __name__ == '__main__':
    states = ['b', 'be', 'b1', 'b2']
    # first_release_df = time_first_release()
    # first_release_df.to_excel('time_first_release_df.xlsx')
    # first_release_df = pL_first_release()
    # first_release_df.to_excel('pL_release_b_states_df.xlsx')
    # load the dataframe
    first_release_df = pd.read_excel('pL_release_b_states_df.xlsx')

    # exclude nan values
    first_release_df = first_release_df[first_release_df['pL'].notna()]

    food_DataFrame = pd.read_excel(home + '\\DataFrame\\food_in_back.xlsx')
    first_release_df = pd.merge(first_release_df, food_DataFrame[['filename', 'food in back']], on='filename',
                                how='left')
    first_release_df['food in back'] = first_release_df['food in back'] != "no"

    sorted = first_release_df
    sorted['pL'] = first_release_df['pL'].apply(lambda lst: None if len(eval(lst)) == 0 else max(eval(lst)))
    sorted = sorted.sort_values(by='pL', ascending=False)

    fig, axs = plt.subplots(1, 4, sharey=True, sharex=True)
    for size, ax in zip(['XL', 'L', 'M', 'S (> 1)', ], axs):
        group = first_release_df[first_release_df['size'] == size]
        # split by food in back
        food_groups = group.groupby(['food in back'])
        for state, food_group in food_groups:
            DEBUG = 1
            # if size == 'S (> 1)' and state == 'ab':
            #     # exclude 'S_SPT_4740002_SSpecialT_1_ants (part 1)' because it is an outlier
            #     group = group[group['filename'] != 'S_SPT_4740002_SSpecialT_1_ants (part 1)']
            # if there are values that are not nan
            values = [eval(l) for l in food_group['pL']]
            flattened_list = [element for sublist in values for element in sublist]
            # make histogram with bars next to each other
            z = ax.hist(flattened_list, bins=15, label='food in back: ' + str(state), range=[0, 30], density=True,
                        color={True: 'green', False: 'red'}[state], alpha=0.5)
            # percent which where notnan
        plt.legend()
        values = [eval(l) for l in group['pL']]

        percent = np.sum([len(v) > 0 for v in values]) / len(group['pL'])
        title = size + ', ' + str(round(percent * 100, 1)) + '%'
        # write title vertically
        ax.text(0.5, 1.08, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        # ax.set_ylim([0, 0.35])
    axs[0].set_ylabel('probability density')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.05, top=0.9, right=0.95, wspace=0.3, hspace=0.5)

    # save the figure

    # axs[2].set_xlabel('pL passed before being released from ' + str(states) + ' [minimal pL]')
    plt.savefig('results\\state_passage_statistics\\' + 'pL_before_release_b.png')
    # axs[2].set_xlabel('time passed before being released from ' + str(states) + ' [min]')
    # plt.savefig('results\\state_passage_statistics\\' + 'time_before_release_b.png')
    plt.show()
