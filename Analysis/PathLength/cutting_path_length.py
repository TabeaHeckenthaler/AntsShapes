"""
Plot for all sizes of ants
distributions of path lengths for successful and unsuccessful experiments
"""

from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.exp_types import solver_geometry, ResizeFactors
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np
from Analysis.PathLength.PathLength import PathLength
from trajectory_inheritance.get import get
import pandas as pd
import pickle
import os
from Analysis.GeneralFunctions import flatten


class Path_length_cut_off_df(Altered_DataFrame):
    def __init__(self, solver):
        super().__init__()
        self.choose_experiments(solver, 'SPT', geometry=solver_geometry[solver], init_cond='back')
        columns = ['filename', 'winner', 'size', 'communication', 'fps', 'time [s]']
        self.choose_columns(columns)
        self.plot_separately = None
        self.color = None
        self.geometry = None

    def add_dictionary(self, d, time_measure='norm solving time [s]',
                       path_length_measure='penalized path length [length unit]'):
        self.df['path length/minimal path length[]'] = self.df['path length [length unit]'] \
                                                       / self.df['minimal path length [length unit]']
        if time_measure is not None and 'norm' in time_measure:
            self.add_normalized_measure(time_measure)
        if 'norm' in path_length_measure:
            self.add_normalized_measure(path_length_measure)

    def __add__(self, Path_length_cut_off_df2):
        self.n_group_sizes = self.n_group_sizes + Path_length_cut_off_df2.n_group_sizes
        self.plot_separately.update(Path_length_cut_off_df2.plot_separately)
        self.solver = self.solver + '_' + Path_length_cut_off_df2.solver
        self.df = pd.concat([self.df, Path_length_cut_off_df2.df])
        return self

    def add_normalized_measure(self, measure):
        """

        """
        self.df[measure] = self.df[measure.split('norm ')[-1]] / self.df['size'].map(ResizeFactors[self.solver])

    # def add_solving_time(self, separate_data_frames) -> dict:
    #     for key, df in separate_data_frames.items():
    #         separate_data_frames[key]['solving time [s]'] = separate_data_frames[key]['time [s]']
    #     return separate_data_frames

    # def add_normalized_measure(self, separate_data_frames: dict, measure: str) -> dict:
    #     for key, df in separate_data_frames.items():
    #         separate_data_frames[key][measure] = separate_data_frames[key][measure.split('norm ')[-1]] / \
    #                                              separate_data_frames[key]['size'].map(ResizeFactors[self.solver])
    #     return separate_data_frames

    @staticmethod
    def get_maximum_value(dic, measure) -> float:
        df = pd.concat([v2[measure] for v2 in flatten([v1.values() for v1 in [v for v in dic.values()]])])
        return df.max()

    def split_separate_groups(self, df=None):
        if df is None:
            df = self.df
        separate_group_df = pd.DataFrame()
        not_separate_group_df = pd.DataFrame()

        for index, s in df.iterrows():
            if s['size'] in self.plot_separately.keys() and \
                    s['average Carrier Number'] in self.plot_separately[s['size']]:
                separate_group_df = separate_group_df.append(s)
            else:
                not_separate_group_df = not_separate_group_df.append(s)
        return separate_group_df, not_separate_group_df

    def open_figure(self):
        fig, axs = plt.subplots(nrows=self.n_group_sizes, sharex=True)
        fig.subplots_adjust(hspace=0.2)
        plt.show(block=False)
        return fig, axs

    def cut_off_after_time(self, name: str, time_measure: str, path_length_measure: str, max_t=30 * 60):
        """
        :param time_measure:
        :param path_length_measure:
        :param max_t:

        """
        if name + '.pkl' in os.listdir():
            with open(name + '.pkl', 'rb') as file:
                self.df = pickle.load(file)

        else:
            def calc_path_length(exp) -> float:
                x = get(exp['filename'])
                if time_measure == 'norm solving time [s]':
                    frames = [0, x.frame_count_after_solving_time(int(exp['norm maximal time [s]']))]
                else:
                    frames = [0, int(exp['norm maximal time [s]'] * exp['fps'])]
                if 'penalized path length' in path_length_measure:
                    return PathLength(x).calculate_path_length(frames=frames, penalize=True)
                if 'path length' in path_length_measure:
                    if np.isnan(frames[1]):
                        return np.NaN
                    return PathLength(x).calculate_path_length(frames=frames, penalize=False)

            self.df['norm maximal time [s]'] = max_t * self.df['size'].map(ResizeFactors[self.solver])
            self.df[path_length_measure.split('/')[0] + ' [length unit]'] = self.df.progress_apply(calc_path_length, axis=1)
            self.df[path_length_measure.split('/')[0].split(' [length unit]')[0] + '/minimal path length[]'] = \
                self.df[path_length_measure.split('/')[0] + ' [length unit]'] \
                / self.df['minimal path length [length unit]']

            with open(name + '.pkl', 'wb') as file:
                pickle.dump(self.df, file)

        not_successful = ~ self.df['winner']
        measured_overtime = self.df[time_measure] > self.df['norm maximal time [s]']
        exclude = (~ measured_overtime & not_successful)
        print(self.df[exclude])
        self.df.drop(self.df[exclude].index, inplace=True)
        self.df['winner'] = ~ (measured_overtime | not_successful)

    def cut_off_after_path_length(self, path_length_measure, max_path=20):
        not_successful = ~ self.df['winner']
        measured_overpath = self.df[path_length_measure] > max_path
        exclude = (~ measured_overpath & not_successful)
        self.df.drop(self.df[exclude].index, inplace=True)
        self.df['winner'] = ~ (measured_overpath | not_successful)

    def plot_means(self, separate_data_frames, on_xaxis, on_yaxis, ax, marker='.'):
        for size, dfs in separate_data_frames.items():
            if size not in ['M (2)', 'M (1)']:
                for key, df in dfs.items():
                    for part in self.split_separate_groups(df):
                        if not part.empty:
                            part.loc[part['size'].isin(['Small Far', 'Small Near']), 'size'] = 'Small'
                            groups = part.groupby(by=['size'])
                            means = groups.mean()
                            sem = groups.sem()
                            # std = groups.std()

                            means.plot.scatter(x=on_xaxis, y=on_yaxis, xerr=sem[on_xaxis], yerr=sem[on_yaxis],
                                               c=self.color[key],
                                               ax=ax,
                                               marker=marker,
                                               s=150)

                            if len(means) > 0:
                                xs = list(means[on_xaxis])
                                ys = list(means[on_yaxis])
                                for txt, x, y in zip(list(means.index), xs, ys):
                                    ax.annotate(txt, (x, y), fontsize=15)
                            ax.set_ylim(0, max(ax.get_ylim()[1], means[on_yaxis][0]+5))
            ax.legend([{'winner': 'successful', 'looser': 'unsuccessful'}[k] for k in dfs.keys()])

    def plot_path_length_distributions(self, separate_data_frames, path_length_measure, axs, **kwargs):
        pass

    def plot_time_distributions(self, separate_data_frames, axs, measure='solving time [s]'):
        pass

    def add_mean_sem(self, df_sizes, measure, ax, colors, max_num_experiments):
        means = [d[measure].mean() for keys, d in df_sizes.items()]
        sems = [d[measure].sem() for keys, d in df_sizes.items()]
        for mean, sem, color, delta in zip(means, sems, colors, range(len(means))):
            ax.errorbar(x=mean, y=max_num_experiments + 1 + delta, xerr=sem, color=color, fmt='o', linewidth=2,
                        capsize=6)


class Path_length_cut_off_df_human(Path_length_cut_off_df):
    def __init__(self, time_measure='norm solving time [s]',
                 path_length_measure='penalized path length [length unit]'):
        self.solver = 'human'
        super().__init__(self.solver)
        self.n_group_sizes = 5
        self.plot_separately = {'Medium': [2, 1]}
        self.color = {'communication': 'blue', 'non_communication': 'orange'}
        self.geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    def average_participants(self, sizes, df):
        d = df[df['size'].isin(sizes)]

        if sizes[0] in self.plot_separately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_separately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in
         [['Large'], ['Medium'], ['Small Far', 'Small Near']]]

    # def add_solving_time(self, separate_data_frames) -> dict:
    #     for size in separate_data_frames.keys():
    #         for comm in separate_data_frames[size].keys():
    #             separate_data_frames[size][comm]['solving time [s]'] = separate_data_frames[size][comm]['time [s]']
    #     return separate_data_frames
    #
    # def add_normalized_measure(self, separate_data_frames: dict, measure: str) -> dict:
    #     for size, df in separate_data_frames.items():
    #         for comm in separate_data_frames[size].keys():
    #             separate_data_frames[size][comm][measure] = \
    #                 separate_data_frames[size][comm][measure.split('norm ')[-1]] / \
    #                 separate_data_frames[size][comm]['size'].map(ResizeFactors[self.solver])
    #     return separate_data_frames

    def plot_path_length_distributions(self, separate_data_frames, path_length_measure, axs, max_path=18):
        colors = ['blue', 'orange']
        bins = np.arange(0, 9, 0.5)
        kwargs = {'bins': bins, 'histtype': 'bar'}

        for i, (size, df_sizes) in enumerate(separate_data_frames.items()):
            axs[i].hist([d['path length/minimal path length[]'] for d in df_sizes.values()], color=colors, **kwargs)
            axs[i].set_ylabel(size)
            axs[i].legend(df_sizes.keys())

        # axs[-1].legend(['communicating', 'non-communicating'])
        axs[-1].set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        for j in range(len(axs)):
            axs[j].yaxis.set_label_coords(labelx, 0.5)

    def plot_time_distributions(self, separate_data_frames, axs, measure='solving time [s]'):
        colors = ['blue', 'orange']
        bins = np.arange(0, 1250, 100)
        max_num_experiments = 1

        for i, (size, df_sizes) in enumerate(separate_data_frames.items()):
            results = axs[i].hist([d[measure] for keys, d in df_sizes.items()], color=colors, bins=bins)
            axs[i].set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)
            self.add_mean_sem(df_sizes, measure, axs[i], colors, max_num_experiments)

        for ax in axs:
            ax.set_ylim([0, max_num_experiments])
        axs[-1].legend(separate_data_frames[list(separate_data_frames.keys())[0]].keys())
        axs[-1].set_xlabel(measure)

        labelx = -0.05  # axes coords
        for ax in axs:
            ax.set_ylim([0, max_num_experiments + 3])
        axs[-1].yaxis.set_label_coords(labelx, 0.5)


class Path_length_cut_off_df_humanhand(Path_length_cut_off_df):
    def __init__(self, time_measure='norm solving time [s]', path_length_measure='penalized path length [length unit]'):
        self.solver = 'humanhand'
        super().__init__(self.solver, time_measure=time_measure, path_length_measure=path_length_measure)

        self.n_group_sizes = 1
        self.plot_separately = {'': []}
        self.geometry = ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')
        self.color = {'with_eyesight': 'red', 'without_eyesight': 'blue'}

    def plot_time_distributions(self, separate_data_frames, axs, measure='solving time [s]'):
        colors = ['blue', 'orange']
        bins = np.arange(0, 150, 10)
        max_num_experiments = 1
        plt.show(block=False)

        results = axs.hist([d['time [s]'] for key, d in separate_data_frames.items()], bins=bins, color=colors)
        axs.set_ylabel('number of experiments')
        max_num_experiments = max(np.max(results[0]), max_num_experiments)
        self.add_mean_sem(separate_data_frames, measure, axs, colors, max_num_experiments)

        axs.legend(separate_data_frames.keys())
        axs.set_xlabel('time [s]')

        # for j in range(len(axs)):
        axs.set_ylim([0, np.ceil(max_num_experiments) + 0.5])
        # labelx = -0.05  # axes coords

        axs.set_ylim([0, max_num_experiments + 3])
        # axs.yaxis.set_label_coords(labelx, 0.5)

    def average_participants(self, sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in self.plot_separately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_separately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in ['']]

    def plot_path_length_distributions(self, separate_data_frames, path_length_measure, axs, max_path=70):
        colors = ['blue', 'orange']
        bins = range(0, max_path, max_path // 6)

        max_num_experiments = 1
        results = axs.hist([d['path length/minimal path length[]'] for keys, d in separate_data_frames.items()],
                           color=colors, bins=bins)
        # av_Carrier_Number()
        axs.set_xlim(0, max_path)
        max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs.legend(separate_data_frames.keys())
        axs.set_xlabel('path length/minimal path length')

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):
        axs.set_ylim([0, max_num_experiments + 1.5])
        axs.yaxis.set_label_coords(labelx, 0.5)

    def percent_of_solving(self):
        data_frames = self.get_separate_data_frames(self.solver, self.plot_separately)
        percent_of_winning = {}
        for size, dfs in data_frames.items():
            percent_of_winning[size] = len(dfs['winner']) / (len(dfs['looser']) + len(dfs['winner']))


class Path_length_cut_off_df_ant(Path_length_cut_off_df):
    def __init__(self):
        self.solver = 'ant'
        super().__init__(self.solver)
        self.n_group_sizes = 5
        self.plot_separately = {'S': [1]}
        self.color = {'winner': 'green', 'looser': 'red'}
        self.geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')

    # def add_solving_time(self, separate_data_frames) -> dict:
    #     for size in separate_data_frames.keys():
    #         if size not in ['S (> 1)', 'Single (1)']:
    #             for success in separate_data_frames[size].keys():
    #                 separate_data_frames[size][success]['solving time [s]'] = \
    #                     separate_data_frames[size][success]['time [s]']
    #         else:
    #             for success in separate_data_frames[size].keys():
    #                 separate_data_frames[size][success]['solving time [s]'] = \
    #                     separate_data_frames[size][success].progress_apply(lambda x: get(x['filename']).solving_time(),
    #                                                                        axis=1)
    #     return separate_data_frames

    # def add_normalized_measure(self, separate_data_frames: dict, measure: str) -> dict:
    #     for size, df in separate_data_frames.items():
    #         for success in separate_data_frames[size].keys():
    #             separate_data_frames[size][success][measure] = \
    #                 separate_data_frames[size][success][measure.split('norm ')[-1]] / \
    #                 separate_data_frames[size][success]['size'].map(ResizeFactors[self.solver])
    #     return separate_data_frames

    def plot_time_distributions(self, separate_data_frames, axs, measure='solving time [s]'):
        colors = ['green', 'red']
        maximum_value = self.get_maximum_value(separate_data_frames, measure)

        bins = np.arange(0, maximum_value, maximum_value // 12)
        max_num_experiments = 1

        for i, (size, df_sizes) in enumerate(separate_data_frames.items()):
            # df_sizes['winner'].hist(column=['path length/minimal path length[]'], ax=axs[2], bins=bins)
            # df_sizes['looser']['path length/minimal path length[]'].hist(ax=axs[2], bins=bins)
            results = axs[i].hist([d[measure] for keys, d in df_sizes.items()], color=colors, bins=bins)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)
            self.add_mean_sem(df_sizes, measure, axs[i], colors, max_num_experiments)
            axs[i].set_ylabel(size)

        axs[-1].legend(separate_data_frames[list(separate_data_frames.keys())[-1]].keys())
        axs[-1].set_xlabel(measure)

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):

        for ax in axs:
            ax.set_ylim([0, max_num_experiments + 3])

        # axs[-1].set_ylim([0, max_num_experiments + 1.5])
        axs[-1].yaxis.set_label_coords(labelx, 0.5)

        fig = plt.figure()
        percent_of_winning, error = self.percent_of_solving()
        # TODO: Add error bars.
        plt.bar(*zip(*percent_of_winning.items()))
        plt.ylabel('percent of success')
        save_fig(fig, 'percent_solving_ants_cut_time')

    def average_participants(self, sizes, df):
        d = df[df['size'] == sizes]

        if sizes[0] in self.plot_separately.keys():
            d = d[~df['average Carrier Number'].isin(self.plot_separately[sizes[0]])]

        [print(sizes, d['average Carrier Number'].mean()) for sizes in ['XL', 'L', 'M', 'S']]

    def plot_path_length_distributions(self, separate_data_frames, path_length_measure, axs_old, **kwargs):
        fig = plt.gcf()
        colors = ['green', 'red']
        max_path = self.get_maximum_value(separate_data_frames, path_length_measure)
        bins = range(0, int(max_path), int(max_path) // 6)

        num_sizes = len(separate_data_frames.keys())
        gs = fig.add_gridspec(num_sizes, 3)

        axs = [fig.add_subplot(gs[i, 2]) for i in range(0, num_sizes)]
        # [axs[i].set_xticklabels([]) for i in range(num_sizes-1)]
        fig.delaxes(axs_old)

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
        #     rects = axs[i].patches
        #     labels = ["{:.1f}".format(mean) + '+-' + "{:.1f}".format(std) if not np.isnan(mean) else ''
        #               for mean, std in zip(av_Carrier_Numbers_mean, av_Carrier_Numbers_std)]
        #
        #     for rect, label in zip(rects, labels):
        #         height = rect.get_height()
        #         axs[i].text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
        #                     ha='center', va='bottom')

        for i, (size, df_sizes) in enumerate(separate_data_frames.items()):
            results = axs[i].hist([d[path_length_measure] for keys, d in df_sizes.items()], color=colors, bins=bins)
            # av_Carrier_Number()
            axs[i].set_xlim(0, max_path)
            axs[i].set_ylabel(size)

            max_num_experiments = max(np.max(results[0]), max_num_experiments)
            self.add_mean_sem(df_sizes, path_length_measure, axs[i], colors, max_num_experiments)

        axs[-1].legend(['successful', 'unsuccessful'])
        axs[-1].set_xlabel(path_length_measure)

        labelx = -0.05  # axes coords
        for i in range(len(axs)):
            axs[i].set_ylim([0, max_num_experiments + 3])
            axs[i].yaxis.set_label_coords(labelx, 0.5)

    def plot_percent_of_solving(self, separate_data_frames, ax):
        percent_of_winning, error = self.calc_percent_of_solving(separate_data_frames)
        ax.bar(*zip(*percent_of_winning.items()))
        ax.errorbar(list(zip(*percent_of_winning.items()))[0],
                    list(zip(*percent_of_winning.items()))[1],
                    yerr=list(zip(*error.items()))[1], fmt="o", color="r")
        ax.set_ylabel('percent of success')
        ax.set_xlabel('size')

    @staticmethod
    def calc_percent_of_solving(separate_data_frames):
        percent_of_winning, error = {}, {}

        for size, dfs in separate_data_frames.items():
            if (len(dfs['looser']) + len(dfs['winner'])) > 0:
                percent_of_winning[size] = len(dfs['winner']) / (len(dfs['looser']) + len(dfs['winner']))
                error[size] = np.sqrt(percent_of_winning[size] * (1 - percent_of_winning[size]) /
                                      np.sum([len(df) for df in dfs.values()]))

        return percent_of_winning, error


def plot_means():
    shape = 'SPT'
    PLs = [Path_length_cut_off_df_ant, Path_length_cut_off_df_human,
           # Path_length_cut_off_df_humanhand
           ]
    fig, axs = plt.subplots(len(PLs), 1)

    for PL, ax in zip(PLs, axs):
        my_PL = PL()
        my_PL.choose_experiments(my_PL.solver, shape, geometry=my_PL.geometry, init_cond='back')
        raise Error
        my_PL.cut_off_after_path_length(max_path=15)
        my_PL.plot_means_violin(ax)

        # adjust_figure(ax)

    [ax.set_xlabel('') for ax in axs[:-1]]
    [ax.set_ylabel('') for ax in [axs[0], axs[-1]]]
    axs[0].set_ylabel('path length/minimal path length[]')

    axs[0].set_ylim(0, 20)
    axs[1].set_ylim(2.5, 4)

    save_fig(fig, 'back_path_length_all')


# def time_distribution(measure='solving time [s]'):
#     Plot_classes = [Path_length_cut_off_df_ant, Path_length_cut_off_df_human, Path_length_cut_off_df_humanhand, ]
#     # Plot_classes = [Path_length_cut_off_df_ant]
#
#     for Plot_class in Plot_classes:
#         print(Plot_class)
#         my_plot_class = Plot_class(measure)
#         separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
#                                                                       my_plot_class.plot_separately)
#
#         fig, axs = my_plot_class.open_figure()
#         my_plot_class.plot_time_distributions(separate_data_frames, axs, measure=measure)
#         save_fig(fig, measure + my_plot_class.solver)
#
#
# def path_length_distribution_after_max_time(time_measure='norm solving time [s]',
#                                             path_length_measure='penalized path length [length unit]'):
#     Plot_classes = [Path_length_cut_off_df_ant]
#     # Plot_classes = [Path_length_cut_off_df_humanhand,
#     #                 Path_length_cut_off_df_ant,
#     #                 Path_length_cut_off_df_human]
#
#     with open('ps.pkl', 'rb') as file:
#         separate_data_frames = pickle.load(file)
#     my_plot_class = Path_length_cut_off_df_ant()
#     fig, axs = my_plot_class.open_figure()
#     my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs)
#     save_fig(fig, 'back_time_' + my_plot_class.solver + 'cut_of_time')
#
#     for Plot_class in Plot_classes:
#         my_plot_class = Plot_class(time_measure=time_measure, path_length_measure=path_length_measure)
#         my_plot_class.cut_off_after_time(time_measure, path_length_measure, seconds_max=30*60)
#         separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
#                                                                       my_plot_class.plot_separately)
#
#         with open('ps.pkl', 'wb') as file:
#             pickle.dump(separate_data_frames, file)
#
#         fig, axs = my_plot_class.open_figure()
#         my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs)
#         save_fig(fig, 'back_time_' + my_plot_class.solver + 'cut_of_time')
#
#
# def path_length_distribution_after_max_path_length(path_length_measure='penalized path length', max_path=15, ax=None):
#     Plot_classes = [Path_length_cut_off_df_ant, Path_length_cut_off_df_human, Path_length_cut_off_df_humanhand]
#
#     measure = ''
#
#     for Plot_class in Plot_classes:
#         my_plot_class = Plot_class(measure)
#         fig, axs = my_plot_class.open_figure()
#         my_plot_class.cut_off_after_path_length(max_path=max_path)
#         separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
#                                                                       my_plot_class.plot_separately)
#         my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs)
#         save_fig(fig, 'back_path_length_' + str(max_path) + my_plot_class.solver + 'cut_of_path')


if __name__ == '__main__':
    # plot_means()
    # path_length_distribution_after_max_time(time_measure='norm time [s]',
    #                                         path_length_measure='penalized path length [length unit]')

    # time_distribution(measure='norm solving time [s]')
    DEBUG = 1

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
    #                                                               my_plot_class.plot_separately)
    # my_plot_class.plot_path_length_distributions(separate_data_frames, axs, max_path=12 + 4)
    # save_fig(fig, 'back_path_length_' + my_plot_class.solver + 'cut_of_path')
