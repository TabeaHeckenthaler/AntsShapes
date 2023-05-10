import pandas as pd
from colors import colors_state
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import dfs_ant
from DataFrame.plot_dataframe import save_fig
import os
import json
from Directories import network_dir
import numpy as np

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()


def fraction_of_time():
    fig, axs = plt.subplots(1, len(dfs_ant.keys()))

    # iterate over all dfs
    for (size, df), ax in zip(dfs_ant.items(), axs):
        # df = df[df['winner']]
        df_percentages = pd.DataFrame(index=df['filename'],
                                      columns=['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g',
                                               'h'])

        # import the time series of all experiments
        df['time series'] = df['filename'].map(time_series_dict)

        # calculate the time spent in each state for every row
        for filename, ts in zip(df['filename'], df['time series']):
            # add the percentages to the df_percentages
            df_percentages.loc[filename] = pd.Series(ts).value_counts(normalize=True)

        means, std = {}, {}
        # iterate over all columns
        for column in df_percentages.columns:
            # drop the nan values
            percentages = df_percentages[column].fillna(0)
            # calculate the mean
            means[column] = percentages.mean()
            std[column] = percentages.std() / np.sqrt(len(percentages))

        # combine means and std to a dataframe
        state_percentages = pd.DataFrame({'mean': pd.Series(means), 'std': pd.Series(std)})

        # plot the mean time spent in each state
        state_percentages.fillna(0, inplace=True)
        # drop column 'g'
        state_percentages.drop('g', inplace=True)
        # sort columns
        state_percentages = state_percentages.reindex(
            ['b', 'be', 'b1', 'b2', 'ab', 'ac', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'h'])

        # plot state_percentages_mean empty bars
        ax.bar(state_percentages.index, state_percentages['mean'].values,
               yerr=state_percentages['std'],
               color=[colors_state[state] for state in state_percentages.index],
               edgecolor=[colors_state[state] for state in state_percentages.index])
        DEBUG = 1
        # set the y-axis limits to 0 to 0.25
        ax.set_ylim(0, 0.25)
        ax.set_title(size)

    fig.set_size_inches(20.3, 4)
    save_fig(fig, 'state_percentages_winner', size)
    DEBUG = 1

