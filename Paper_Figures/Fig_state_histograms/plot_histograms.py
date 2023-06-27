import matplotlib.pyplot as plt
import pandas as pd
from colors import colors_state
from DataFrame.import_excel_dfs import df_ant_excluded
import json
from os import path
from Directories import network_dir, home

date = '2023_06_27'

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(path.join(home, 'DataFrame\\frameNum_dict.json'), 'r') as json_file:
    frameNum_dict = json.load(json_file)

df_ant_excluded['time series'] = df_ant_excluded['filename'].map(time_series_dict)
df_ant_excluded['frameNum'] = df_ant_excluded['filename'].map(frameNum_dict)
ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']
df_ant_excluded = df_ant_excluded[df_ant_excluded['winner']]

df_sim = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
with open(path.join(home, 'Gillespie\\' + date + '_sim_time_series.json'), 'r') as json_file:
    time_series_sim_dict = json.load(json_file)
    json_file.close()
df_sim['time series'] = df_sim['filename'].map(time_series_sim_dict)

DEBUG = 1


def plot_state_percentages(df, ax, **kwargs):
    # find the percentage of every state in the state series
    states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'b1/b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
    state_percentages = pd.DataFrame(0, index=df['filename'], columns=states)
    for filename, ts in zip(df['filename'], df['time series']):
        state_percentages.loc[filename] = pd.Series(ts).value_counts(normalize=True)

    state_percentages['b1/b2'] = (state_percentages['b1'] + state_percentages['b2'])/2
    # change position column 'b1/b2' to come after 'ac'
    state_percentages.drop(columns=['b1', 'b2'], inplace=True)

    state_percentages.fillna(0, inplace=True)

    # find the average percentage of every state
    state_percentages_mean = state_percentages.mean()

    # plot state_percentages_mean empty bars
    ax.bar(state_percentages_mean.index, state_percentages_mean.values,
           color=[colors_state[state] for state in state_percentages_mean.index],
           edgecolor=[colors_state[state] for state in state_percentages_mean.index], **kwargs)


if __name__ == '__main__':
    name = {'Single (1)': 'single', 'S (> 1)': 'SMany', 'M': 'M', 'L': 'L', 'XL': 'XL'}

    ant_sizes = ['XS', 'S', 'M', 'L', 'XL']
    size_trans = {'XL': 'XL', 'M': 'M', 'L': 'L', 'S (> 1)': 'S'}

    df_ant_excluded['size'] = df_ant_excluded['size'].map(size_trans)

    fig, axs = plt.subplots(1, len(ant_sizes), figsize=(len(ant_sizes)*5, 5), sharey=True)
    for size, ax in zip(ant_sizes, axs):
        df_exp_size = df_ant_excluded[df_ant_excluded['size'] == size]
        df_sim_size = df_sim[df_sim['size'] == size]
        plot_state_percentages(df_exp_size, ax, alpha=0.3, label='experiment')
        plot_state_percentages(df_sim_size, ax, alpha=1, label='simulation', fill=False, linewidth=2)
        ax.set_title(size + ',  only successful')
        plt.legend()
    fig.tight_layout()

    plt.savefig('histograms\\histograms' + date + '.png')
    plt.savefig('histograms\\histograms' + date + '.eps')
    plt.savefig('histograms\\histograms' + date + '.svg')
