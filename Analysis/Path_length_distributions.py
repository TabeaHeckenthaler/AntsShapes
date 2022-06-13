"""
Plot for all sizes of ants
distributions of path lengths for successful and unsuccessful experiments
"""

from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.exp_types import solver_geometry, ResizeFactors
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np
from Analysis.PathLength import PathLength
from trajectory_inheritance.get import get
import pandas as pd
from DataFrame.plot_dataframe import reduce_legend
from Analysis.GeneralFunctions import flatten


class Path_length_cut_off_df(Altered_DataFrame):
    def __init__(self, solver):
        super().__init__()
        self.choose_experiments(solver, 'SPT', geometry=solver_geometry[solver], init_cond='back')

        columns = ['filename', 'winner', 'size', 'communication', 'path length [length unit]',
                   'minimal path length [length unit]', 'average Carrier Number', 'time [s]', 'fps', ]
        self.choose_columns(columns)

        self.df['path length/minimal path length[]'] = self.df['path length [length unit]'] \
                                                       / self.df['minimal path length [length unit]']
        self.plot_seperately = None
        self.color = None
        self.geometry = None

    def __add__(self, Path_length_cut_off_df2):
        self.n_group_sizes = self.n_group_sizes + Path_length_cut_off_df2.n_group_sizes
        self.plot_seperately.update(Path_length_cut_off_df2.plot_seperately)
        self.solver = self.solver + '_' + Path_length_cut_off_df2.solver
        self.df = pd.concat([self.df, Path_length_cut_off_df2.df])
        return self

    def split_seperate_groups(self, df=None):
        if df is None:
            df = self.df
        seperate_group_df = pd.DataFrame()
        not_seperate_group_df = pd.DataFrame()

        for index, s in df.iterrows():
            if s['size'] in self.plot_seperately.keys() and \
                    s['average Carrier Number'] in self.plot_seperately[s['size']]:
                seperate_group_df = seperate_group_df.append(s)
            else:
                not_seperate_group_df = not_seperate_group_df.append(s)
        return seperate_group_df, not_seperate_group_df

    def open_figure(self):
        fig, axs = plt.subplots(nrows=self.n_group_sizes, sharex=True)
        fig.subplots_adjust(hspace=0.2)
        plt.show(block=False)
        return fig, axs

    def cut_off_after_time(self, seconds_max=30 * 60):
        self.df['maximal time [s]'] = seconds_max  # * self.df['size'].map(ResizeFactors[self.solver])
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

    def plot_means(self, ax, marker='.'):
        d = self.get_separate_data_frames(self.solver, self.plot_seperately, 'SPT', self.geometry)

        for size, dfs in d.items():
            for key, df in dfs.items():
                for part in self.split_seperate_groups(df):
                    if not part.empty:
                        part.loc[part['size'].isin(['Small Far', 'Small Near']), 'size'] = 'Small'
                        groups = part.groupby(by=['size'])
                        means = groups.mean()
                        sem = groups.sem()
                        # std = groups.std()

                        means.plot.scatter(x='average Carrier Number',
                                           y='path length/minimal path length[]',
                                           xerr=sem['average Carrier Number'],
                                           yerr=sem['path length/minimal path length[]'],
                                           c=self.color[key],
                                           ax=ax,
                                           marker=marker,
                                           s=150)

                        if len(means) > 0:
                            xs = list(means['average Carrier Number'] + 0.05)
                            ys = list(means['path length/minimal path length[]'] + 0.05)
                            for txt, x, y in zip(list(means.index), xs, ys):
                                ax.annotate(txt, (x, y), fontsize=8)
        ax.legend(dfs.keys())
        ax.set_title(self.solver)

    def plot_path_length_distributions(self, seperate_data_frames, axs, **kwargs):
        pass

    def plot_time_distributions(self, seperate_data_frames, axs):
        pass


class Path_length_cut_off_df_human(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'human'
        super().__init__(self.solver)
        self.n_group_sizes = 5
        self.plot_seperately = {'Medium': [2, 1]}
        self.color = {'communication': 'blue', 'non_communication': 'orange'}
        self.geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    def average_participants(self, sizes, df):
        d = df[df['size'].isin(sizes)]

        if sizes[0] in self.plot_seperately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_seperately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in
         [['Large'], ['Medium'], ['Small Far', 'Small Near']]]

    def plot_path_length_distributions(self, seperate_data_frames, axs, max_path=18):
        colors = ['blue', 'orange']
        bins = np.arange(0, 9, 0.5)
        kwargs = {'bins': bins, 'histtype': 'bar'}

        for i, (size, df_sizes) in enumerate(seperate_data_frames.items()):
            axs[i].hist([d['path length/minimal path length[]'] for d in df_sizes.values()], color=colors, **kwargs)
            axs[i].set_ylabel(size)
            axs[i].legend(df_sizes.keys())

        # axs[-1].legend(['communicating', 'non-communicating'])
        axs[-1].set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        for j in range(len(axs)):
            axs[j].yaxis.set_label_coords(labelx, 0.5)

    def plot_time_distributions(self, seperate_data_frames, axs):
        colors = ['green', 'red']
        bins = np.arange(0, 1250, 100)
        max_num_experiments = 1

        for i, (size, df_sizes) in enumerate(seperate_data_frames.items()):
            results = axs[i].hist([d['time [s]'] for keys, d in df_sizes.items()], color=colors, bins=bins)
            axs[i].set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs[-1].legend(seperate_data_frames[list(seperate_data_frames.keys())[0]].keys())
        axs[-1].set_xlabel('time [s]')

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):
        # axs.set_ylim([0, max_num_experiments + 1.5])
        axs[-1].yaxis.set_label_coords(labelx, 0.5)


class Path_length_cut_off_df_humanhand(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'humanhand'
        super().__init__(self.solver)

        self.n_group_sizes = 1
        self.plot_seperately = {'': []}
        self.geometry = ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')
        self.color = {'with_eyesight': 'red', 'without_eyesight': 'blue'}

    def plot_time_distributions(self, seperate_data_frames, axs):
        bins = np.arange(0, 150, 10)
        max_num_experiments = 1
        plt.show(block=False)

        results = axs.hist([d['time [s]'] for key, d in seperate_data_frames.items()], bins=bins)
        axs.set_ylabel('number of experiments')
        max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs.legend(seperate_data_frames.keys())
        axs.set_xlabel('time [s]')

        # for j in range(len(axs)):
        axs.set_ylim([0, np.ceil(max_num_experiments)+0.5])
        # labelx = -0.05  # axes coords
        # axs.yaxis.set_label_coords(labelx, 0.5)

    def average_participants(self, sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in self.plot_seperately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_seperately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in ['']]

    def plot_path_length_distributions(self, seperate_data_frames, axs, max_path=70):
        colors = ['green', 'red']
        bins = range(0, max_path, max_path // 6)

        max_num_experiments = 1

        # def av_Carrier_Number():
        #     av_Carrier_Numbers_mean, av_Carrier_Numbers_std = [], []
        #     for (key, df), boundaries in zip(df_sizes.items(), results[0]):
        #         hey = df.sort_values(by='path length/minimal path length[]')
        #         boundaries_hist = [0] + np.cumsum(boundaries).tolist()
        #
        #         av_Carrier_Numbers_mean += [hey.iloc[int(b1):int(b2)]['average Carrier Number'].mean()
        #                                     for b1, b2 in zip(boundaries_hist[:-1], boundaries_hist[1:])]
        #
        #         av_Carrier_Numbers_std += [hey.iloc[int(b1):int(b2)]['average Carrier Number'].std()
        #                                    for b1, b2 in zip(boundaries_hist[:-1], boundaries_hist[1:])]
        #
        #     # Make some labels.
        #     rects = axs.patches
        #     labels = ["{:.1f}".format(mean) + '+-' + "{:.1f}".format(std) if not np.isnan(mean) else ''
        #               for mean, std in zip(av_Carrier_Numbers_mean, av_Carrier_Numbers_std)]
        #
        #     for rect, label in zip(rects, labels):
        #         height = rect.get_height()
        #         axs.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
        #                  ha='center', va='bottom')

        for i, (size, df_sizes) in enumerate(seperate_data_frames.items()):
            # df_sizes['winner'].hist(column=['path length/minimal path length[]'], ax=axs[2], bins=bins)
            # df_sizes['looser']['path length/minimal path length[]'].hist(ax=axs[2], bins=bins)
            results = axs.hist([d['path length/minimal path length[]'] for keys, d in df_sizes.items()],
                               color=colors, bins=bins)
            # av_Carrier_Number()
            axs.set_xlim(0, max_path)
            axs.set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs.legend(seperate_data_frames[''].keys())
        axs.set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):
        axs.set_ylim([0, max_num_experiments + 1.5])
        axs.yaxis.set_label_coords(labelx, 0.5)

    def percent_of_solving(self):
        data_frames = self.get_separate_data_frames(self.solver, self.plot_seperately)
        percent_of_winning = {}
        for size, dfs in data_frames.items():
            percent_of_winning[size] = len(dfs['winner']) / (len(dfs['looser']) + len(dfs['winner']))


class Path_length_cut_off_df_ant(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'ant'
        super().__init__(self.solver)

        self.n_group_sizes = 5
        self.plot_seperately = {'S': [1]}
        self.color = {'winner': 'green', 'looser': 'red'}
        self.geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')

    def plot_time_distributions(self, seperate_data_frames, axs):
        colors = ['green', 'red']
        bins = np.arange(0, 1250, 100)
        max_num_experiments = 1

        for i, (size, df_sizes) in enumerate(seperate_data_frames.items()):
            # df_sizes['winner'].hist(column=['path length/minimal path length[]'], ax=axs[2], bins=bins)
            # df_sizes['looser']['path length/minimal path length[]'].hist(ax=axs[2], bins=bins)
            results = axs[i].hist([d['time [s]'] for keys, d in df_sizes.items()], color=colors, bins=bins)
            axs[i].set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs[-1].legend(seperate_data_frames[list(seperate_data_frames.keys())[-1]].keys())
        axs[-1].set_xlabel('time [s]')

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):
        axs[-1].set_ylim([0, max_num_experiments + 1.5])
        axs[-1].yaxis.set_label_coords(labelx, 0.5)

        fig = plt.figure()
        percent_of_winning, error = self.percent_of_solving()
        # TODO: Add error bars.
        plt.bar(*zip(*percent_of_winning.items()))
        plt.ylabel('percent of success')
        save_fig(fig, 'percent_solving_ants_cut_time')

    def average_participants(self, sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in self.plot_seperately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_seperately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in ['XL', 'L', 'M', 'S']]

    def plot_path_length_distributions(self, seperate_data_frames, axs, max_path=70):
        colors = ['green', 'red']
        bins = range(0, max_path, max_path // 6)

        max_num_experiments = 1

        def av_Carrier_Number():
            av_Carrier_Numbers_mean, av_Carrier_Numbers_std = [], []
            for (key, df), boundaries in zip(df_sizes.items(), results[0]):
                hey = df.sort_values(by='path length/minimal path length[]')
                boundaries_hist = [0] + np.cumsum(boundaries).tolist()

                av_Carrier_Numbers_mean += [hey.iloc[int(b1):int(b2)]['average Carrier Number'].mean()
                                            for b1, b2 in zip(boundaries_hist[:-1], boundaries_hist[1:])]

                av_Carrier_Numbers_std += [hey.iloc[int(b1):int(b2)]['average Carrier Number'].std()
                                           for b1, b2 in zip(boundaries_hist[:-1], boundaries_hist[1:])]

            # Make some labels.
            rects = axs[i].patches
            labels = ["{:.1f}".format(mean) + '+-' + "{:.1f}".format(std) if not np.isnan(mean) else ''
                      for mean, std in zip(av_Carrier_Numbers_mean, av_Carrier_Numbers_std)]

            for rect, label in zip(rects, labels):
                height = rect.get_height()
                axs[i].text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                            ha='center', va='bottom')

        for i, (size, df_sizes) in enumerate(seperate_data_frames.items()):
            # df_sizes['winner'].hist(column=['path length/minimal path length[]'], ax=axs[2], bins=bins)
            # df_sizes['looser']['path length/minimal path length[]'].hist(ax=axs[2], bins=bins)
            results = axs[i].hist([d['path length/minimal path length[]'] for keys, d in df_sizes.items()],
                                  color=colors, bins=bins)
            av_Carrier_Number()
            axs[i].set_xlim(0, max_path)
            axs[i].set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs[-1].legend(['successful', 'unsuccessful'])
        axs[-1].set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        for j in range(len(axs)):
            axs[j].set_ylim([0, max_num_experiments + 1.5])
            axs[j].yaxis.set_label_coords(labelx, 0.5)

        fig = plt.figure()
        percent_of_winning, error = self.percent_of_solving()
        # TODO: Add error bars.
        plt.bar(*zip(*percent_of_winning.items()))
        plt.ylabel('percent of success')
        save_fig(fig, 'percent_solving_ants_cut_time')

    def percent_of_solving(self):
        data_frames = self.get_separate_data_frames(self.solver, self.plot_seperately)
        percent_of_winning, error = {}, {}

        for size, dfs in data_frames.items():
            percent_of_winning[size] = len(dfs['winner']) / (len(dfs['looser']) + len(dfs['winner']))
            error[size] = np.sqrt(percent_of_winning[size] * (1-percent_of_winning[size])/
                                  np.sum([len(df) for df in dfs.values()]))

        return percent_of_winning, error


def plot_means():
    shape = 'SPT'
    PLs = [Path_length_cut_off_df_ant,
           # Path_length_cut_off_df_human, Path_length_cut_off_df_humanhand
           ]
    fig, axs = plt.subplots(len(PLs), 1)

    for PL, ax in zip(PLs, axs):
        my_PL = PL()
        my_PL.choose_experiments(my_PL.solver, shape, geometry=my_PL.geometry, init_cond='back')
        # columns = ['filename', 'winner', 'size', 'communication', 'path length [length unit]',
        #            'minimal path length [length unit]',
        #            'average Carrier Number']
        # df.choose_columns(columns)
        my_PL.cut_off_after_path_length(max_path=15)
        my_PL.plot_means(ax)

        # adjust_figure(ax)

    [ax.set_xlabel('') for ax in axs[:-1]]
    [ax.set_ylabel('') for ax in [axs[0], axs[-1]]]
    save_fig(fig, 'back_path_length_all')


def cut_time():
    # Plot_classes = [Path_length_cut_off_df_human, Path_length_cut_off_df_ant, Path_length_cut_off_df_humanhand]
    Plot_classes = [Path_length_cut_off_df_humanhand, Path_length_cut_off_df_ant, Path_length_cut_off_df_human]

    for Plot_class in Plot_classes:
        my_plot_class = Plot_class()
        fig, axs = my_plot_class.open_figure()
        separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
                                                                      my_plot_class.plot_seperately)
        # my_plot_class.cut_off_after_time()
        # my_plot_class.plot_path_length_distributions(separate_data_frames, axs, max_path=25)
        my_plot_class.plot_time_distributions(separate_data_frames, axs)
        save_fig(fig, 'back_time_' + my_plot_class.solver + 'cut_of_time')


def cut_path_length_distribution(max_path=15, ax=None):
    Plot_classes = [Path_length_cut_off_df_ant,
                    # Path_length_cut_off_df_human, Path_length_cut_off_df_humanhand
                    ]

    for Plot_class in Plot_classes:
        my_plot_class = Plot_class()
        fig, axs = my_plot_class.open_figure()
        my_plot_class.cut_off_after_path_length(max_path=max_path)
        separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
                                                                      my_plot_class.plot_seperately)
        my_plot_class.plot_path_length_distributions(separate_data_frames, axs, max_path=max_path)
        save_fig(fig, 'back_path_length_' + str(max_path) + my_plot_class.solver + 'cut_of_path')


def percent_of_solving_ants(max_path=15, ax=None):
    my_plot_class = Path_length_cut_off_df_ant()
    my_plot_class.cut_off_after_path_length(max_path=max_path)
    percent_of_winning, error = my_plot_class.percent_of_solving()

    if ax is not None:
        ax = plt.figure()
        ax.bar(*zip(*percent_of_winning.items()), yerr=error.values())
        ax.set_title('min_path ' + str(max_path))
        ax.set_ylim([-0.5, 1.2])
        plt.ylabel('percent of success')

    return percent_of_winning, error


if __name__ == '__main__':
    cut_time()

    # max_paths = list(range(10, 26, 2))
    # fig, axs = plt.subplots(nrows=len(max_paths)//2, ncols=2, sharey=True, sharex=True)
    # fig = plt.figure()
    # percent_for_cutoffs, errors = {}, {}

    # for ax, max_path in zip(flatten(axs), max_paths):
    # for max_path in max_paths:
    #     percent_for_cutoffs[max_path], errors[max_path] = percent_of_solving_ants(max_path=max_path)

    # d = pd.DataFrame(percent_for_cutoffs)
    # e = pd.DataFrame(errors)
    # d.transpose().plot(yerr=e.transpose(), capsize=4, capthick=1)
    #
    # save_fig(plt.gcf(), 'percent_solving_ants_cut_path_all')

    # my_plot_class = Path_length_cut_off_df_human() + Path_length_cut_off_df_humanhand()
    # fig, axs = my_plot_class.open_figure()
    # separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
    #                                                               my_plot_class.plot_seperately)
    # my_plot_class.plot_path_length_distributions(separate_data_frames, axs, max_path=12 + 4)
    # save_fig(fig, 'back_path_length_' + my_plot_class.solver + 'cut_of_path')
