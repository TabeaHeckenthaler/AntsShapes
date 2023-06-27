import matplotlib.pyplot as plt
import pandas as pd
from colors import colors_state
from DataFrame.import_excel_dfs import df_ant_excluded
import json
from os import path
from Directories import network_dir, home
from itertools import groupby
import numpy as np

date = '2023_06_27'

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(path.join(home, 'DataFrame\\frameNum_dict.json'), 'r') as json_file:
    frameNum_dict = json.load(json_file)

df_ant_excluded['time series'] = df_ant_excluded['filename'].map(time_series_dict)
df_ant_excluded['frameNum'] = df_ant_excluded['filename'].map(frameNum_dict)
ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:]
df_ant_excluded = df_ant_excluded[df_ant_excluded['winner']]


df_sim = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
with open(path.join(home, 'Gillespie\\' + date + '_sim_time_series.json'), 'r') as json_file:
    time_series_sim_dict = json.load(json_file)
    json_file.close()
df_sim['time series'] = df_sim['filename'].map(time_series_sim_dict)


def time_stamped_series(time_series, time_step) -> list:
    groups = groupby(time_series)
    return [(label, sum(1 for _ in group) * time_step) for label, group in groups]


def calc_durations(df, size_list, name='durations' + date + '.json'):
    durations = {}
    fig, axs = plt.subplots(1, len(size_list), figsize=(len(size_list) * 4, 5), sharey=True)
    for size, ax in zip(size_list, axs):
        df_size = df[df['size'] == size]
        durations[size] = {}
        for i, row in df_size.iterrows():
            time_series = row['time series']

            # replace all 'cg' with 'c'
            # time_series = [state if state != 'cg' else 'c' for state in time_series]

            time_step = 0.25
            time_stamped = time_stamped_series(time_series, time_step)

            for state, duration in time_stamped:
                if state not in durations[size].keys():
                    durations[size][state] = []
                durations[size][state].append(duration)
    # save to json file
    with open(name, 'w') as json_file:
        json.dump(durations, json_file)
        json_file.close()


def find_the_longest_durations(df):
    durations = {}
    size_list = 'S (> 1)'
    for size in zip(size_list):
        df_size = df[df['size'] == size]
        durations[size] = {}
        for i, row in df_size.iterrows():
            time_series = row['time series']

            # replace all 'cg' with 'c'
            # time_series = [state if state != 'cg' else 'c' for state in time_series]

            time_step = 0.25
            time_stamped = time_stamped_series(time_series, time_step)

            for state, duration in time_stamped:
                if state not in durations[size].keys():
                    durations[size][state] = []
                durations[size][state].append(duration)

    # save to json file
    with open(name, 'w') as json_file:
        json.dump(durations, json_file)
        json_file.close()


if __name__ == '__main__':
    name = {'Single (1)': 'single', 'S (> 1)': 'SMany', 'M': 'M', 'L': 'L', 'XL': 'XL'}

    find_the_longest_durations(df_ant_excluded)

    # calc_durations(df_sim, ['XS', 'S', 'M', 'L'], name='durations_sim' + date + '.json')
    calc_durations(df_ant_excluded, ant_sizes, name='durations_exp.json')

    # # load from json file
    # name = 'sim'
    # with open('durations_sim' + date + '.json', 'r') as json_file:
    #     durations = json.load(json_file)
    # sizes = ['XS', 'S', 'M', 'L']

    name = 'exp'
    with open('durations_exp.json', 'r') as json_file:
        durations = json.load(json_file)
    sizes = ant_sizes

    # states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h', 'i']
    states, range_, ylim = ['c', 'cg'], (0, 1), (0, 20)
    # states = ['c'], (0, 10)

    axs = {}
    figs = {}
    for state in states:
        figs[state], axs[state] = plt.subplots(1, len(sizes), figsize=(20, 5), sharey=True)

    for i in range(len(sizes)):
        size = sizes[i]
        for state in states:
            axs[state][i].hist(np.array(durations[size][state])/60, bins=20, density=True, alpha=0.5,
                               color=colors_state[state], range=range_, label=name + ' ' + state)
            # axs[state][i].hist(np.array(durations_sim[size][state])/60, bins=20, density=True, alpha=0.5,
            #                    color=colors_state[state], range=(0, 1), label='simulation ' + state, fill=False,
            #                    linewidth=2)

        for state in states:
            axs[state][i].set_title(size + ', only successful')
            axs[state][i].set_xlabel('duration [min]')
            axs[state][i].set_ylabel('probability density')
            axs[state][i].legend()
            axs[state][i].set_ylim(ylim)

    for state in states:
        figs[state].savefig(name + date + f'_duration_in_state_distribution_{state}.png')
    DEBUG = 1