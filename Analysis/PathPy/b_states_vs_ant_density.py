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

b_state_df = pd.DataFrame(columns=['filename', 'size', 'b_states_indices', 'ant_density'])
b_state_df['filename'] = df_ant['filename']
b_state_df['size'] = df_ant['size']


def get_indices(time_series) -> list:
    is_in_states = np.isin(time_series, states)
    succession_indices = np.flatnonzero(np.concatenate(([False], np.diff(is_in_states.astype(int)) != 0))).tolist()
    if len(succession_indices) > 0:
        if len(succession_indices) % 2 == 1:
            succession_indices.append(len(time_series) - 1)
        # split the succession_indices into pairs as type list
        pairs = [(succession_indices[i], succession_indices[i + 1]) for i in range(0, len(succession_indices), 2)]
    else:
        pairs = []
    return pairs


def find_ant_number(traj, ind) -> tuple:
    if traj.participants is None:
        traj.load_participants()
    if ind >= len(traj.participants.frames):
        print(traj.filename, ind, ' ind is out of range')
    if traj.participants.frames[ind] is None:
        return None, None
    traj.participants.frames[ind].clean()
    attached, non_attached = traj.participants.frames[ind].count_in_back_room()
    return attached, non_attached


def process_filename(filename):
    print(filename)
    ts = time_series_dict[filename]
    x = get(filename)
    time_series = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)

    b_states = []
    ant_density = []
    probed_frames = []
    pairs = get_indices(time_series)
    for pair in pairs:
        print(pair)
        if np.abs(np.diff(pair)) > 5 * x.fps:
            traj_cut = copy(x).cut_off(frame_indices=pair)
            # print(state)
            # traj.play()
            b_states.append(pair)
            pf = list(np.arange(*pair, 5 * x.fps))
            probed_frames.append(pf)
            ant_density.append([find_ant_number(x, p) for p in pf])
    return b_states, ant_density


def save_error_to_file(error):
    # Function to save the error to a text file
    with open("error_log.txt", "a") as file:
        file.write(error + "\n")


def indices_of_b_states() -> pd.DataFrame:
    b_states_indices_dict = {}
    ant_density_dict = {}
    start = 0
    for i, filename in tqdm(enumerate(df_ant['filename'][start:], start), total=len(df_ant['filename'][start:])):
        print(i)
        # filename = 'L_SPT_4420007_LSpecialT_1_ants (part 1)'
        save_error_to_file(filename + ': ')
        try:
            b_states_indices_dict[filename], ant_density_dict[filename] = process_filename(filename)
        except Exception as e:
            # Save the error to a text file
            save_error_to_file(str(e))
            b_states_indices_dict[filename], ant_density_dict[filename] = None, None
        DEBUG = 1

    b_state_df['b_states_indices'] = b_state_df['filename'].map(b_states_indices_dict)
    b_state_df['ant_density'] = b_state_df['filename'].map(ant_density_dict)
    return b_state_df


def scatter_plot():
    sizes = ['M', 'L', 'XL']
    fig, axs = plt.subplots(1, len(sizes), figsize=(15, 5))
    # iterate over every row (filename)
    for size, ax in zip(sizes, axs):
        for i, row in df[df['size'] == size].iterrows():
            print(row['filename'])

            # if row['filename'] == 'L_SPT_4660018_LSpecialT_1_ants (part 1)':
            #     DEBUG = 1
            DEBUG = 1
            pL = eval(row['pL'])
            if row['ant_density'] is not np.nan:
                ant_counts = eval(row['ant_density'])
                mask = [(ant_count[0][0] is not None) and (ant_count[1][0] is not None) for ant_count in ant_counts]
                pL = np.array(pL)[mask]
                ant_counts = np.array(ant_counts)[mask]
                ant_counts = [np.array(ant_count).sum() / len(ant_count) for ant_count in ant_counts]
                ax.scatter(ant_counts, pL, label=row['filename'])
            else:
                ant_counts = None
            ax.set_ylabel('pL [minimal path length]]')
            ax.set_xlabel('ant counts from tracking')
            ax.set_title(size)
            # plt.legend()\
    plt.tight_layout()
    plt.savefig('pL_vs_ant_counts.png')
    plt.show()


def get_averages(row):
    y = eval(row['pL'])
    ant_counts = eval(row['ant_density'])
    mask = [~np.any(np.array(ant_count) == None) for ant_count in ant_counts]
    x = [xi for xi, m in zip(ant_counts, mask) if m]
    y = [yi for yi, m in zip(y, mask) if m]
    new_x = [np.array(xi).sum() / len(xi) for xi in x]
    if 'XL' in row['filename'] and np.any(np.array(new_x) < 100) and np.any(np.array(y) < 2):
        DEBUG = 1
    return new_x, y


def binned_plot():
    limits = {'M': [15, 75], 'L': [15, 80], 'XL': [25, 250]}

    num_bins = 10
    sizes = ['M', 'L', 'XL']
    fig, axs = plt.subplots(1, len(sizes), figsize=(15, 5))
    # iterate over every row (filename)
    for size, ax in zip(sizes, axs):
        ys = []
        xs = []
        for i, row in df[df['size'] == size].iterrows():
            print(row['filename'])
            if row['filename'] == 'M_SPT_4700010_MSpecialT_1_ants (part 1)':
                DEBUG = 1

            if row['ant_density'] is not np.nan:
                x, y = get_averages(row)
                xs += x
                ys += y

        # bin_width = (np.max(xs) - np.min(xs)) / num_bins
        # bins = np.arange(np.min(xs), np.max(xs) + bin_width, bin_width)

        # Compute the bin edges
        bin_width = (limits[size][1] - limits[size][0]) / num_bins
        bins = np.arange(limits[size][0], limits[size][1] + bin_width, bin_width)[:num_bins + 1]
        xs, ys = np.array(xs), np.array(ys)
        # Compute the mean of y for each bin
        bin_means = []
        for i in range(1, num_bins + 1):
            mask = (xs >= bins[i - 1]) & (xs <= bins[i])
            bin_data = ys[mask]
            bin_mean = np.mean(bin_data)
            bin_means.append(bin_mean)

        # Plot the scatter plot
        # draw vertical lines at the bin edges
        for bin in bins:
            ax.axvline(x=bin, color='k', linestyle='--', alpha=0.2)
        ax.scatter(xs, ys, alpha=0.5, label='Data Points')
        ax.set_ylabel('pL [minimal path length]]')
        ax.set_xlabel('ant counts from tracking')
        # Plot the mean of each bin
        ax.plot(bins[:-1] + bin_width / 2, bin_means, 'ro-', label='Bin Means')
        ax.set_title(size)
    plt.savefig('pL_vs_ant_counts_binning.png')


def save_list_of_lower_ant_numbers():
    to_low = {'M': 12.5, 'L': 35, 'XL': 70}
    sizes = ['M', 'L', 'XL']
    to_exclude = []
    # iterate over every row (filename)
    for size in sizes:
        for i, row in df[df['size'] == size].iterrows():
            if row['ant_density'] is not np.nan:
                x, y = get_averages(row)
                if np.any(np.array(x) < to_low[size]):
                    print(row['filename'])
                    print(x)
                    print(y)
                    print('------------------')
                    to_exclude.append(row['filename'])
    print(to_exclude)
    to_exclude = ['M_SPT_4690001_MSpecialT_1_ants', 'M_SPT_4690011_MSpecialT_1_ants (part 1)',
                  'L_SPT_4650012_LSpecialT_1_ants (part 1)', 'L_SPT_4660014_LSpecialT_1_ants',
                  'L_SPT_4670008_LSpecialT_1_ants (part 1)', 'L_SPT_4420007_LSpecialT_1_ants (part 1)',
                  'L_SPT_4420004_LSpecialT_1_ants', 'L_SPT_4420005_LSpecialT_1_ants (part 1)',
                  'L_SPT_4420010_LSpecialT_1_ants (part 1)', 'L_SPT_5030001_LSpecialT_1_ants (part 1)',
                  'L_SPT_5030009_LSpecialT_1_ants (part 1)', 'L_SPT_5030006_LSpecialT_1_ants (part 1)',
                  'XL_SPT_4640001_XLSpecialT_1_ants (part 1)', 'XL_SPT_5040006_XLSpecialT_1_ants (part 1)',
                  'XL_SPT_5040012_XLSpecialT_1_ants', 'XL_SPT_5040003_XLSpecialT_1_ants (part 1)']


if __name__ == '__main__':
    states = ['b', 'be', 'b1', 'b2']
    # b_state_df = indices_of_b_states()
    # b_state_df.to_excel('b_states_indices_ant_counts.xlsx')

    pL_release_df = pd.read_excel('pL_release_b_states_df.xlsx')
    b_state_df = pd.read_excel('b_states_indices_ant_counts.xlsx')

    df = pd.merge(pL_release_df, b_state_df, on=['filename', 'size'], how='outer')

    df_L = df[df['size'] == 'L']
    # binned_plot()
    save_list_of_lower_ant_numbers()

    DEBUG = 1
