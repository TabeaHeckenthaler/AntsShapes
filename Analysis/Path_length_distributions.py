"""
Plot for all sizes of ants
distributions of path lengths for successful and unsuccessful experiments
"""

from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.trajectory import solver_geometry
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np

color = {'ant': {0: 'black', 1: 'black'}, 'human': {0: 'red', 1: 'blue'}}
plot_group_size_seperately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}}


def plot_path_length_distributions_human(df, axs):
    sep = plot_group_size_seperately['human']
    colors = ['blue', 'orange']
    bins = np.arange(0, 9, 0.5)
    kwargs = {'bins': bins, 'histtype': 'bar'}

    def data_frames(sizes, df):
        communication = df[df['communication'] & (df['size'].isin(sizes))]['path length/minimal path length[]']
        non_communication = df[~df['communication'] & (df['size'].isin(sizes))]['path length/minimal path length[]']
        return [communication, non_communication]

    def average_participants(sizes, df):
        d = df[df['size'].isin(sizes)]

        if sizes[0] in sep.keys():
            d = d[~df['average Carrier Number'].isin(sep[sizes[0]])]

        return d['average Carrier Number'].mean()

    [print(sizes, average_participants(sizes, df)) for sizes in [['Large'], ['Medium'], ['Small Far', 'Small Near']]]

    axs[0].hist(data_frames(['Large'], df), color=colors, **kwargs)

    axs[1].hist([dataframe[~df['average Carrier Number'].isin(sep['Medium'])] for dataframe in data_frames(['Medium'], df)],
                color=colors, **kwargs)
    axs[2].hist([dataframe[df['average Carrier Number'].isin([sep['Medium'][0]])] for dataframe in data_frames(['Medium'], df)],
                color=colors, **kwargs)
    axs[3].hist([dataframe[df['average Carrier Number'].isin([sep['Medium'][1]])] for dataframe in data_frames(['Medium'], df)],
                color=colors, **kwargs)
    axs[4].hist(data_frames(['Small Far', 'Small Near'], df), color=colors, **kwargs)

    for ax, label in zip(axs, ['L', 'M', 'M', 'M', 'S']):
        ax.set_xlim(0, np.max(bins))
        ax.set_ylabel(label)
    axs[-1].legend(['communicating', 'non-communicating'])
    axs[-1].set_xlabel('path length/minimal path length')
    labelx = -0.05  # axes coords

    for j in range(len(axs)):
        axs[j].yaxis.set_label_coords(labelx, 0.5)


def plot_path_length_distributions_ant(df, axs):
    sep = plot_group_size_seperately['ant']
    colors = ['green', 'red']
    bins = range(0, 70, 3)

    def average_participants(sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in sep.keys():
            d = d[~df['average Carrier Number'].isin(sep[sizes[0]])]

        return d['average Carrier Number'].mean()

    [print(sizes, average_participants(sizes, df)) for sizes in ['XL', 'L', 'M', 'S']]

    def data_frames(size, df):
        winner = df[df['winner'] & (df['size'] == size)]['path length/minimal path length[]']
        looser = df[~df['winner'] & (df['size'] == size)]['path length/minimal path length[]']
        return [winner, looser]

    for i, size in enumerate(['XL', 'L', 'M']):
        axs[i].hist(data_frames(size, df), color=colors, bins=bins)
        axs[i].set_ylabel(size)

    axs[3].hist([data_frame[~df['average Carrier Number'].isin(sep['S'])]
                 for data_frame in data_frames('S', df)], color=colors, bins=bins)
    axs[3].set_ylabel('S')

    axs[4].hist([data_frame[df['average Carrier Number'].isin(sep['S'])]
                 for data_frame in data_frames('S', df)], color=colors, bins=bins)
    axs[4].set_ylabel('Single')

    # m = max([ax.get_xlim()[1] for ax in axs])
    # [ax.set_xlim(0, m) for ax in axs]
    axs[-1].legend(['successful', 'unsuccessfull'])
    axs[-1].set_xlabel('path length/minimal path length')

    labelx = -0.05  # axes coords
    for j in range(len(axs)):
        axs[j].yaxis.set_label_coords(labelx, 0.5)


if __name__ == '__main__':
    shape = 'SPT'

    solvers = ['ant', 'human']
    image_names = ['back_path_length_ants', 'back_path_length_humans']
    plot_functions = [plot_path_length_distributions_ant, plot_path_length_distributions_human]
    n_rows = {'human': 5, 'ant': 5}

    for solver, image_name, plot_function in zip(solvers, image_names, plot_functions):
        df = Altered_DataFrame()
        df.choose_experiments(solver, shape, solver_geometry[solver], init_cond='back')

        columns = ['filename', 'winner', 'size', 'communication', 'path length [length unit]',
                   'minimal path length [length unit]', 'average Carrier Number']
        df.choose_columns(columns)

        df.df['path length/minimal path length[]'] = df.df['path length [length unit]'] \
                                                     / df.df['minimal path length [length unit]']

        fig, axs = plt.subplots(nrows=n_rows[solver], sharex=True)
        fig.subplots_adjust(hspace=0.2)
        plt.show(block=False)
        plot_function(df.df, axs)
        save_fig(fig, image_name)
