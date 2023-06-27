import matplotlib.pyplot as plt
import pandas as pd
from colors import colors_state
from DataFrame.import_excel_dfs import df_ant_excluded
import json
from os import path
from Directories import network_dir, home
from itertools import groupby
from trajectory_inheritance.get import get
from tqdm import tqdm
import numpy as np

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(path.join(home, 'DataFrame\\frameNum_dict.json'), 'r') as json_file:
    frameNum_dict = json.load(json_file)

df_ant_excluded['time series'] = df_ant_excluded['filename'].map(time_series_dict)
df_ant_excluded['frameNum'] = df_ant_excluded['filename'].map(frameNum_dict)
ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:]
df_ant_excluded = df_ant_excluded[df_ant_excluded['winner']]

df_sim = pd.read_excel(home + '\\Gillespie\\2023_06_20_sim.xlsx')
with open(path.join(home, 'Gillespie\\2023_06_20_sim_time_series.json'), 'r') as json_file:
    time_series_sim_dict = json.load(json_file)
    json_file.close()
df_sim['time series'] = df_sim['filename'].map(time_series_sim_dict)
sim_sizes = ['XS', 'S', 'M', 'L']


def extend_time_series_to_match_frames(ts, frame_len):
    indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                         for _ in range(frame_len)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def save_vel_in_c_and_cg_exp():
    vel_in_cg, vel_in_c = {}, {}

    # sort by size
    for size in ant_sizes:
        vel_in_cg[size], vel_in_c[size] = {}, {}
        df = df_ant_excluded[df_ant_excluded['size'] == size]
        for i, row in tqdm(df.iterrows(), total=len(df), desc=size):
            filename = row['filename']
            time_series = row['time series']
            x = get(filename)
            frameNum = len(x.frames)
            time_series = extend_time_series_to_match_frames(time_series, frameNum)
            vel = np.linalg.norm(x.velocity(smoothed=1), axis=1)

            indices_c = [i for i, state in enumerate(time_series) if state == 'c']
            vel_in_c[size][filename] = [vel[i] for i in indices_c]
            # x.play(ts=time_series)

            indices_cg = [i for i, state in enumerate(time_series) if state == 'cg']
            vel_in_cg[size][filename] = [vel[i] for i in indices_cg]

            # # plot velocity
            # plt.figure()
            # plt.plot(vel, alpha=0.5)
            # plt.plot(indices_c, [vel[i] for i in indices_c],  color='red', alpha=0.5)
            # plt.plot(indices_cg, [vel[i] for i in indices_cg], color='green', alpha=0.5)

    # save to json
    with open('vel_in_c_exp_smoothed1s.json', 'w') as json_file:
        json.dump(vel_in_c, json_file)
        json_file.close()

    with open('vel_in_cg_exp_smoothed1s.json', 'w') as json_file:
        json.dump(vel_in_cg, json_file)
        json_file.close()


def save_vel_in_c_and_cg_sim():
    vel_in_cg, vel_in_c = {}, {}

    # sort by size
    for size in sim_sizes:
        vel_in_cg[size], vel_in_c[size] = {}, {}
        df = df_sim[df_sim['size'] == size]
        for i, row in tqdm(df.iterrows(), total=len(df), desc=size):
            filename = row['filename']
            time_series = row['time series']
            x = get(filename)
            x.adapt_fps(50)
            frameNum = len(x.frames)
            time_series = extend_time_series_to_match_frames(time_series, frameNum)
            vel = np.linalg.norm(x.velocity(smoothed=False), axis=1)

            indices_c = [i for i, state in enumerate(time_series) if state == 'c']
            vel_in_c[size][filename] = [vel[i] for i in indices_c]

            indices_cg = [i for i, state in enumerate(time_series) if state == 'cg']
            vel_in_cg[size][filename] = [vel[i] for i in indices_cg]

    # save to json
    with open(path.join(home, 'DataFrame\\vel_in_c_sim.json'), 'w') as json_file:
        json.dump(vel_in_c, json_file)
        json_file.close()

    with open(path.join(home, 'DataFrame\\vel_in_cg_sim.json'), 'w') as json_file:
        json.dump(vel_in_cg, json_file)
        json_file.close()


def plot_vel_histograms(exp, sim, ymax=20):
    fig_exp, axs = plt.subplots(1, len(ant_sizes), figsize=(20, 5))
    for size, ax in zip(ant_sizes, axs):
        values = [*exp[size].values()]
        values = [item for sublist in values for item in sublist]
        ax.hist(values, bins=40, alpha=0.5, label=size + ' exp', range=(0, 2), density=True)
        ax.set_title(size)
        ax.legend()
        ax.set_ylim(0, ymax)

    fig_sim, axs = plt.subplots(1, len(ant_sizes), figsize=(20, 5))
    for size, ax in zip(sim_sizes, axs):
        values = [*sim[size].values()]
        values = [item for sublist in values for item in sublist]
        ax.hist(values, bins=40, alpha=0.5, label=size + ' sim ', range=(0, 2), density=True)
        ax.set_title(size)
        ax.set_ylim(0, ymax)
        ax.legend()

    return fig_exp, fig_sim


if __name__ == '__main__':
    # save_vel_in_c_and_cg_sim()
    # save_vel_in_c_and_cg_exp()

    type = 'unsmoothed'

    with open('vel_in_c_sim.json', 'r') as json_file:
        vel_in_c_sim = json.load(json_file)
        json_file.close()

    with open('vel_in_c_exp_unsmoothed.json', 'r') as json_file:
        vel_in_c_exp = json.load(json_file)
        json_file.close()

    fig_exp, fig_sim = plot_vel_histograms(vel_in_c_exp, vel_in_c_sim, ymax=23)
    fig_exp.savefig('vel_in_c_exp.png')
    fig_sim.savefig('vel_in_c_sim.png')

    with open('vel_in_cg_sim.json', 'r') as json_file:
        vel_in_cg_sim = json.load(json_file)
        json_file.close()

    with open('vel_in_cg_exp_unsmoothed.json', 'r') as json_file:
        vel_in_cg_exp = json.load(json_file)
        json_file.close()

    fig_exp, fig_sim = plot_vel_histograms(vel_in_cg_exp, vel_in_cg_sim, ymax=23)
    fig_exp.savefig('vel_in_cg_exp.png')
    fig_sim.savefig('vel_in_cg_sim.png')
