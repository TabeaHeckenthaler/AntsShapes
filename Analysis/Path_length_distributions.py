"""
Plot for all sizes of ants
distributions of path lengths for successful and unsuccessful experiments
"""

from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.trajectory import solver_geometry
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np
from Analysis.PathLength import PathLength
from trajectory_inheritance.trajectory import get

ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'human': {'Small Near': 0.25, 'Small Far': 0.25, 'Medium': 0.5, 'Large': 1},
                 'humanhand': {'': 1}}

color = {'ant': {0: 'black', 1: 'black'}, 'human': {0: 'red', 1: 'blue'}}


class Path_length_cut_off_df(Altered_DataFrame):
    def __init__(self, solver):
        super().__init__()
        self.choose_experiments(solver, 'SPT', solver_geometry[solver], init_cond='back')

        columns = ['filename', 'winner', 'size', 'communication', 'path length [length unit]',
                   'minimal path length [length unit]', 'average Carrier Number', 'time [s]', 'fps']
        self.choose_columns(columns)

        self.df['path length/minimal path length[]'] = self.df['path length [length unit]'] \
                                                       / self.df['minimal path length [length unit]']

    def open_figure(self):
        fig, axs = plt.subplots(nrows=my_plot_class.n_group_sizes, sharex=True)
        fig.subplots_adjust(hspace=0.2)
        plt.show(block=False)
        return fig, axs

    def cut_off_after_time(self, seconds_max=30 * 60):
        self.df['maximal time [s]'] = seconds_max * self.df['size'].map(ResizeFactors[self.solver])
        not_successful = ~ self.df['winner']
        measured_overtime = self.df['time [s]'] > self.df['maximal time [s]']
        exclude = (~ measured_overtime & not_successful)
        self.df.drop(self.df[exclude].index, inplace=True)
        self.df['winner'] = ~ (measured_overtime | not_successful)

        def path_length(exp) -> float:
            x = get(exp['filename'])
            frames = [0, int(exp['maximal time [s]'] * exp['fps'])]
            return PathLength(x).calculate_path_length(frames=frames)

        self.df['path length [length unit]'] = \
            self.df.progress_apply(path_length, axis=1)

        self.df['path length/minimal path length[]'] = self.df['path length [length unit]'] \
                                                       / self.df['minimal path length [length unit]']

    def cut_off_after_path_length(self, max_path=15):
        self.df['maximal path length [length unit]'] = max_path * self.df['minimal path length [length unit]']
        not_successful = ~ self.df['winner']
        measured_overpath = self.df['path length [length unit]'] > self.df['maximal path length [length unit]']
        exclude = (~ measured_overpath & not_successful)
        self.df.drop(self.df[exclude].index, inplace=True)
        self.df['winner'] = ~ (measured_overpath | not_successful)

        def path_length(exp) -> float:
            # x = get(exp['filename'])
            # return min(PathLength(x).calculate_path_length(), exp['maximal path length [length unit]'])
            return min(exp['path length [length unit]'], exp['maximal path length [length unit]'])

        self.df['path length [length unit]'] = self.df.progress_apply(path_length, axis=1)
        self.df['path length/minimal path length[]'] = self.df['path length [length unit]'] \
                                                       / self.df['minimal path length [length unit]']

    def plot_path_length_distributions(self, axs, **kwargs):
        pass


class Path_length_cut_off_df_human(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'human'
        super().__init__(self.solver)
        self.n_group_sizes = 5
        self.plot_seperately = {'Medium': [2, 1]}

    def average_participants(self, sizes, df):
        d = df[df['size'].isin(sizes)]

        if sizes[0] in self.plot_seperately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_seperately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in
         [['Large'], ['Medium'], ['Small Far', 'Small Near']]]

    def plot_path_length_distributions(self, axs, max_path=25):
        colors = ['blue', 'orange']
        bins = np.arange(0, 9, 0.5)
        kwargs = {'bins': bins, 'histtype': 'bar'}

        for i, (size, df_sizes) in enumerate(self.get_separate_data_frames(self.solver, self.plot_seperately).items()):
            axs[i].hist([d['path length/minimal path length[]'] for d in df_sizes], color=colors, **kwargs)
            axs[i].set_ylabel(size)

        axs[-1].legend(['communicating', 'non-communicating'])
        axs[-1].set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        for j in range(len(axs)):
            axs[j].yaxis.set_label_coords(labelx, 0.5)


class Path_length_cut_off_df_ant(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'ant'
        super().__init__(self.solver)

        self.n_group_sizes = 5
        self.plot_seperately = {'S': [1]}

    def average_participants(self, sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in self.plot_seperately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_seperately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in ['XL', 'L', 'M', 'S']]

    def plot_path_length_distributions(self, axs, max_path=70):
        colors = ['green', 'red']
        bins = range(0, max_path, max_path//6)

        for i, (size, df_sizes) in enumerate(self.get_separate_data_frames(self.solver, self.plot_seperately).items()):
            axs[i].hist([d['path length/minimal path length[]'] for d in df_sizes], color=colors, bins=bins)
            axs[i].set_ylabel(size)

        axs[-1].legend(['successful', 'unsuccessful'])
        axs[-1].set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        for j in range(len(axs)):
            axs[j].yaxis.set_label_coords(labelx, 0.5)

    def percent_of_solving(self, fig):
        data_frames = self.get_separate_data_frames(self.solver, self.plot_seperately)
        percent_of_winning = {}
        for size, dfs in data_frames.items():
            percent_of_winning[size] = len(dfs[0])/(len(dfs[0]) + len(dfs[1]))
        plt.bar(*zip(*percent_of_winning.items()))
        plt.ylabel('percent of success')
        pass


if __name__ == '__main__':
    shape = 'SPT'

    # Plot_classes = [Path_length_cut_off_df_human, Path_length_cut_off_df_ant]
    Plot_classes = [Path_length_cut_off_df_ant]

    for Plot_class in Plot_classes:
        my_plot_class = Plot_class()
        fig, axs = my_plot_class.open_figure()
        my_plot_class.cut_off_after_time()
        my_plot_class.plot_path_length_distributions(axs, max_path=25)
        save_fig(fig, 'back_path_length_' + my_plot_class.solver + 'cut_of_time')

        if my_plot_class.solver == 'ant':
            fig = plt.figure()
            my_plot_class.percent_of_solving(fig)
            save_fig(fig, 'percent_solving_ants_cut_time')

        my_plot_class = Plot_class()
        fig, axs = my_plot_class.open_figure()
        max_path = 18
        my_plot_class.cut_off_after_path_length(max_path=max_path)
        my_plot_class.plot_path_length_distributions(axs, max_path=max_path+1)
        save_fig(fig, 'back_path_length_' + str(max_path) + my_plot_class.solver + 'cut_of_path')

        if my_plot_class.solver == 'ant':
            fig = plt.figure()
            my_plot_class.percent_of_solving(fig)
            save_fig(fig, 'percent_solving_ants_cut_path')
