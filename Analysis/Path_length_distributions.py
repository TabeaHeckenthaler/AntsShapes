"""
Plot for all sizes of ants
distributions of path lengths for successful and unsuccessful experiments
"""

from matplotlib import pyplot as plt
import numpy as np
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.trajectory import get, solver_geometry
from DataFrame.plot_dataframe import reduce_legend
from DataFrame.dataFrame import myDataFrame as df
from DataFrame.dataFrame import choose_relevant_experiments
import json
from Analysis.PathLength import PathLength
import os
from trajectory_inheritance.trajectory import sizes

color = {'ant': {0: 'black', 1: 'black'}, 'human': {0: 'red', 1: 'blue'}}
plot_group_size_seperately = {'ant': [1], 'human': [2]}


def plot_path_length_distributions(df, solver, axs):

    sep = plot_group_size_seperately[solver]

    for i, size in enumerate(['XL', 'L', 'M']):
        df[~df['winner'] & (df['size'] == size)]['path length/minimal path length[]'].plot.hist(ax=axs[i], alpha=0.5)
        df[df['winner'] & (df['size'] == size)]['path length/minimal path length[]'].plot.hist(ax=axs[i], alpha=0.5)
        axs[i].set_ylabel(size)

    df[~df['winner'] & (df['size'] == 'S')][~df['average Carrier Number'].isin(sep)]['path length/minimal path length[]'].plot.hist(ax=axs[3], alpha=0.5)
    df[df['winner'] & (df['size'] == 'S')][~df['average Carrier Number'].isin(sep)]['path length/minimal path length[]'].plot.hist(ax=axs[3], alpha=0.5)
    axs[3].set_ylabel('S')

    df[~df['winner'] & (df['size'] == 'S') & df['average Carrier Number'].isin(sep)]['path length/minimal path length[]'].plot.hist(ax=axs[4], alpha=0.5)
    df[df['winner'] & (df['size'] == 'S') & df['average Carrier Number'].isin(sep)]['path length/minimal path length[]'].plot.hist(ax=axs[4], alpha=0.5)
    axs[4].set_ylabel('Single')

    m = max([ax.get_xlim()[1] for ax in axs])
    [ax.set_xlim(0, m) for ax in axs]
    axs[-1].legend(['unsuccessfull', 'successful'])
    axs[-1].set_xlabel('path length/minimal path length')


def relevant_columns(df):
    columns = ['filename', 'winner', 'size', 'communication', 'path length [length unit]',
               'minimal path length [length unit]', 'average Carrier Number']
    df = df[columns]
    df['path length/minimal path length[]'] = df['path length [length unit]'] / df['minimal path length [length unit]']
    return df


# def adjust_figure():
#     ax.set_ylim(0, 50)
#     ax.set_xscale('log')
#     # ax.set_yscale('log')
#     # ax.set_ylim(0, 25)
#     plt.show()


if __name__ == '__main__':
    solver = 'ant'
    shape = 'SPT'

    df_relevant_exp = choose_relevant_experiments(df.clone(), shape, solver, solver_geometry[solver], init_cond='back')
    relevant_df = relevant_columns(df_relevant_exp)
    fig, axs = plt.subplots(nrows=len(relevant_df['size'].unique()) + len(plot_group_size_seperately[solver]))
    plt.show(block=False)
    plot_path_length_distributions(relevant_df, solver, axs)

    # adjust_figure()
    save_fig(fig, 'back_path_length_humans')
