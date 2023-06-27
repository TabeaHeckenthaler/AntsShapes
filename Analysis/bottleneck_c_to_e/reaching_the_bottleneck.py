import pandas as pd
from matplotlib import pyplot as plt
from Directories import home, network_dir
import json
from os import path
import numpy as np
from tqdm import tqdm
from DataFrame.import_excel_dfs import df_ant_excluded

plt.rcParams["font.family"] = "Times New Roman"

# save in json
with open(path.join(home, 'DataFrame\\frameNum_dict.json'), 'r') as json_file:
    frameNum_dict = json.load(json_file)

df_ant_excluded['frameNum'] = df_ant_excluded['filename'].map(frameNum_dict)
ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']
with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()


def extend_time_series_to_match_frames(num_frames: int, ts: list) -> list:
    indices_to_ts_to_frames = np.cumsum([1 / (int(num_frames / len(ts) * 10) / 10)
                                         for _ in range(num_frames)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def save_all_indices():
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts.xlsx',
                                                index_col=0)
    # group bottleneck_passing_attempts by 'filename'
    filename_groups = bottleneck_passing_attempts.groupby('filename')

    # dataframe with entrance and exit indices for all filenames
    df = pd.DataFrame(columns=['filename', 'entrance_inds', 'exit_inds', 'bottleneck_inds_start',
                               'bottleneck_inds_end', 'frameNum', 'fps'])

    # iterate over every row in df_ant_excluded
    for index, row in tqdm(df_ant_excluded.iterrows()):
        # get filename
        filename = row['filename']
        ts = extend_time_series_to_match_frames(int(row['frameNum']), time_series_dict[filename])
        # replace all 'cg' with 'c'
        ts = ['c' if state == 'cg' else state for state in ts]

        # find all indices in which 'c' was entered and 'c' was not the previous state
        entrance_inds = [i for i in range(1, len(ts)) if ts[i] == 'c' and ts[i - 1] != 'c']

        # find all indices in which 'c' was left and 'c' was not the next state
        exit_inds = [i for i in range(len(ts) - 1) if ts[i] == 'c' and ts[i + 1] != 'c']

        if filename not in filename_groups.groups.keys():
            bottleneck_inds_start = []
            bottleneck_inds_end = []

        else:
            group = filename_groups.get_group(filename)
            bottleneck_inds_start = group['start'].tolist()
            bottleneck_inds_end = group['end'].tolist()

        # note in df
        df = df.append({'filename': filename, 'entrance_inds': entrance_inds,
                        'exit_inds': exit_inds, 'bottleneck_inds_start': bottleneck_inds_start,
                        'bottleneck_inds_end': bottleneck_inds_end, 'frameNum': row['frameNum'],
                        'fps': row['fps']}, ignore_index=True)

    # save df
    df.to_excel(folder + 'c_g_bottleneck_phasespace\\c_to_bottleneck_indices.xlsx')


def rates():
    df = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\c_to_bottleneck_indices.xlsx')
    # drop 'Unnamed: 0' column
    df = df.drop(columns=['Unnamed: 0'])

    # go through all columns and convert strings to lists
    for col in df.columns:
        if col not in ['filename', 'frameNum', 'fps']:
            df[col] = df[col].map(eval)

    df_new = pd.DataFrame(columns=['filename', 'entrance_ind', 'exit_ind', 'bottleneck_entrance_indices',
                                   'bottleneck_exit_indices', 'frameNum', 'fps', 'time_in_c', 'time_in_bottleneck',
                                   'entered_bottleneck'])

    # save_all_indices()
    # iterate over every row in df_ant_excluded
    for index, row in tqdm(df.iterrows()):
        # time until entered bottleneck
        # how many times was the bottleneck entered during visit to c?
        for i in range(len(row['entrance_inds'])):
            if len(row['exit_inds']) > i:
                print(row['filename'])
                bottleneck_indices = (np.array(row['bottleneck_inds_start']) > row['entrance_inds'][i]) & \
                                     (np.array(row['bottleneck_inds_start']) < row['exit_inds'][i])

                entered_bottleneck = np.sum(bottleneck_indices)

                bottleneck_entrance_indices = np.array(row['bottleneck_inds_start'])[bottleneck_indices]
                bottleneck_exit_indices = np.array(row['bottleneck_inds_end'])[bottleneck_indices]

                time_in_c = (row['exit_inds'][i] - row['entrance_inds'][i]) / row['fps']
                time_in_bottleneck = np.sum((bottleneck_exit_indices - bottleneck_entrance_indices) / row['fps'])

                df_new = df_new.append({'filename': row['filename'], 'entrance_ind': row['entrance_inds'][i],
                                        'exit_ind': row['exit_inds'][i],
                                        'bottleneck_entrance_indices': bottleneck_entrance_indices,
                                        'bottleneck_exit_indices': bottleneck_exit_indices, 'frameNum': row['frameNum'],
                                        'fps': row['fps'], 'time_in_c': time_in_c, 'time_in_bottleneck': time_in_bottleneck,
                                        'entered_bottleneck': entered_bottleneck}, ignore_index=True)
    # save to excel
    df_new.to_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_sep_by_etr_to_c.xlsx')
    DEBUG = 1


def get_statistics():
    df = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_sep_by_etr_to_c.xlsx')
    Num_entrances = {}
    time_in_bottleneck = {}

    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 5))
    for size in ant_sizes:
        filenames = df_ant_excluded[df_ant_excluded['size'] == size]['filename'].tolist()
        df_size = df[df['filename'].isin(filenames)]

        # mean number of entrances to bottleneck
        Num_entrances[size] = df_size['entered_bottleneck'].tolist()
        # fraction of time in bottleneck
        time_in_bottleneck[size] = (df_size['time_in_bottleneck']/df_size['time_in_c']).tolist()

    # save to Num_entrances to json
    with open(folder + 'c_g_bottleneck_phasespace\\Num_entrances.json', 'w') as fp:
        json.dump(Num_entrances, fp)

    # save to time_in_bottleneck to json
    with open(folder + 'c_g_bottleneck_phasespace\\time_in_bottleneck.json', 'w') as fp:
        json.dump(time_in_bottleneck, fp)


def plot_statistics():
    fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54 * 1.5, 5))

    # load from json
    with open(folder + 'c_g_bottleneck_phasespace\\Num_entrances.json', 'r') as fp:
        Num_entrances = json.load(fp)

    with open(folder + 'c_g_bottleneck_phasespace\\time_in_bottleneck.json', 'r') as fp:
        time_in_bottleneck = json.load(fp)

    # plot the mean
    for size in ant_sizes:
        n = Num_entrances[size]
        t = time_in_bottleneck[size]
        axs[0].bar(size, np.mean(n), yerr=np.std(n)/np.sqrt(len(n)), capsize=5, color='lightskyblue')
        axs[1].bar(size, np.mean(t), yerr=np.std(t)/np.sqrt(len(t)), capsize=5, color='lightskyblue')

    axs[0].set_ylabel('Number of entrances to bottleneck')
    axs[1].set_ylabel('Fraction of time in bottleneck')
    DEBUG = 1

    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts.xlsx',
                                                index_col=0)
    size_groups = bottleneck_passing_attempts.groupby('size')

    # find percentage of winner in every size group
    percentage_winner = {}
    std = {}
    for size in ant_sizes:
        group = size_groups.get_group(size)
        percentage_winner[size] = group['winner'].sum() / len(group)
        std[size] = np.sqrt(percentage_winner[size] * (1 - percentage_winner[size]) / len(group))

    # plot the percentage of winner in every size group
    # sort the dictionary by ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    percentage_winner = {size: percentage_winner[size] for size in sizes}

    axs[2].bar(ant_sizes, [percentage_winner[size] for size in ant_sizes],
               yerr=[std[size] for size in ant_sizes], capsize=5,
               color='lightskyblue')
    axs[2].set_ylabel('bottleneck passage probability')
    axs[2].set_xlabel('size')
    plt.tight_layout()

    plt.show()
    DEBUG = 1


if __name__ == '__main__':
    folder = home + '\\Analysis\\bottleneck_c_to_e\\results\\percentage_around_corner\\'
    # save_all_indices()
    # rates()
    # get_statistics()
    plot_statistics()