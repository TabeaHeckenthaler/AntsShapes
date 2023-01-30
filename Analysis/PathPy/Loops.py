from Analysis.PathPy.Path import time_series_dict, time_series_dict_selected_states_sim
from DataFrame.Altered_DataFrame import Altered_DataFrame
from DataFrame.dataFrame import myDataFrame, myDataFrame_sim
from trajectory_inheritance.exp_types import exp_types, solver_geometry
from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from ConfigSpace.ConfigSpace_SelectedStates import perfect_states
from itertools import groupby
import numpy as np
from Analysis.PathPy.Path import color_dict
import pandas as pd
from Analysis.GeneralFunctions import flatten

font = {'family': 'Times New Roman',
        # 'weight' : 'bold',
        'size': 22}

plt.rc('font', **font)

# plt.rcParams["font.family"] = "Times New Roman"

cond_to_include = {'ant': {'XL': ['winner', 'looser'],
                           'L': ['winner', 'looser'],
                           'M': ['winner', 'looser'],
                           'S (> 1)': ['winner', 'looser'],
                           'Single (1)': ['looser']},
                   'human': {'Large': ['communication', 'non_communication'],
                             'M (>7)': ['communication', 'non_communication'],
                             'Small': ['non_communication']},
                   'gillespie': {'M': ['']}
                   }

cond_to_include_split_comm = {'ant': {'XL': ['winner', 'looser'],
                                      'L': ['winner', 'looser'],
                                      'M': ['winner', 'looser'],
                                      'S (> 1)': ['winner', 'looser'],
                                      'Single (1)': ['looser']},
                              'human': {'Large communication': [''],
                                        'M (>7) communication': [''],
                                        'Large non_communication': [''],
                                        'M (>7) non_communication': [''],
                                        'Small non_communication': ['']},
                              'gillespie': {'M': ['']}
                              }

# excluded = ['ant Single (1) winner', 'Small communication', 'M (2) communication', 'M (2) non_communication',
# 'M (1) communication', 'M (1) non_communication']


myDataFrame_sim = Altered_DataFrame(myDataFrame_sim)
myDataFrame_sim.df['time series'] = myDataFrame_sim.df['filename'].map(time_series_dict_selected_states_sim)
filenames_sim = {'gillespie M': myDataFrame_sim.df['filename'].tolist()}

myDataFrame = Altered_DataFrame(myDataFrame)
myDataFrame.choose_experiments(shape='SPT', init_cond='back', free=False, geometry='new')
myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict)
filenames = myDataFrame.get_seperate_filenames()

filenames.update(filenames_sim)

colors = {'ant XL winner': 'green',
          'ant XL looser': 'red',
          'ant L winner': 'green',
          'ant L looser': 'red',
          'ant M winner': 'green',
          'ant M looser': 'red',
          'ant S (> 1) winner': 'green',
          'ant S (> 1) looser': 'red',
          'ant Single (1) looser': 'red',
          'human Large communication': 'blue',
          'human Large non_communication': 'orange',
          'human M (>7) communication': 'blue',
          'human M (>7) non_communication': 'orange',
          'human Small non_communication': 'orange',
          'gillespie M': 'grey'}

label_abbreviations = {'S (> 1)': 'S (> 1 ants)',
                       'Single (1)': 'S (1 ant)',
                       'Large communication': 'Large C',
                       'M (>7) communication': 'Medium C',
                       'Large non_communication': 'Large NC',
                       'M (>7) non_communication': 'Medium NC',
                       'Small non_communication': 'Small',
                       }


class Trajectory:
    def __init__(self, filename, time_series, winner):
        self.filename = filename
        self.time_series = time_series
        self.state_series = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        self.loops = None
        self.winner = winner

    def find_loops(self):
        self.loops = [[]]
        for s in self.state_series:
            if s not in perfect_states:
                self.loops[-1].append(s)
            elif len(self.loops[-1]) != 0:
                self.loops.append([])
        self.loops = list(filter(None, self.loops))

    def count_frequency(self):
        # Creating an empty dictionary
        freq = {}
        for items in self.loops:
            freq[str(items)] = self.loops.count(items)

    def percent_before_passage(self, state):
        if state not in self.time_series:
            return np.NaN
        return np.where(np.array(self.time_series) == state)[0][0] / len(self.time_series)

    def percent_spent_in_state(self, state):
        return len(np.where(np.array(self.time_series) == state)[0]) / len(self.time_series)

    def count_cg(self):
        return self.loops.count(['cg'])

    def count_b1_b2_loops(self):
        return min(np.sum(np.array(self.state_series) == 'b2'), np.sum(np.array(self.state_series) == 'b1'))

    def count_length_in_trap(self, trap: list):
        condition = np.isin(self.state_series, trap)
        list = [sum(1 for _ in group) for key, group in groupby(condition) if key]
        if 'ab' in trap and self.state_series[0] == 'ab':
            list = list[1:]
        return list


def find_traps(df):
    df['repetition of c loop'] = df['filename'].map(
        {t.filename: t.count_length_in_trap(['c', 'cg']) for t in ts})

    df['repetition of b loop'] = df['filename'].map(
        {t.filename: t.count_length_in_trap(['b', 'be', 'b1', 'b2']) for t in ts})

    df['repetition of e loop'] = df['filename'].map(
        {t.filename: t.count_length_in_trap(['e', 'eb', 'eg']) for t in ts})
    return df


def find_rotations(df):
    df['repetition of ac ab loop'] = df['filename'].map(
        {t.filename: t.count_length_in_trap(['ab', 'ac']) for t in ts})
    return df


def find_number_of_transitions(df_original, cut_off=1000, conds_plot_sep=cond_to_include_split_comm):
    num_of_trans = {t.filename: len(t.state_series) for t in ts}
    df_original['number of transitions'] = df_original['filename'].map(num_of_trans)

    not_successful = ~ df_original['winner']
    measured_over_number_of_transitions = df_original['number of transitions'] > cut_off
    exclude = (~ measured_over_number_of_transitions & not_successful)

    df = df_original.drop(df_original[exclude].index)
    df['winner'] = ~ (measured_over_number_of_transitions | not_successful)

    winners_per_exp_type = {}
    for (solver, conds) in conds_plot_sep.items():
        dataframe = pd.DataFrame()
        dataframe['exp_type'] = conds
        fs = {size: [(solver + ' ' + size + ' ' + c).strip() for c in cond_list]
              for size, cond_list in conds.items()}
        filename_sets = {size: flatten([filenames[fi] for fi in f]) for size, f in fs.items()}
        dataframe['filenames'] = dataframe['exp_type'].map(filename_sets)

        winners_per_exp_type[solver] = {}
        for exp_type, series in dataframe.iterrows():
            winners = df[df['filename'].isin(series['filenames'])]['winner']
            if len(winners) > 0:
                winners_per_exp_type[solver][exp_type] = winners.sum() / len(winners)
            else:
                winners_per_exp_type[solver][exp_type] = np.nan

    df_winners = {solver: pd.DataFrame() for solver in winners_per_exp_type.keys()}
    for solver, exp_type_winner_dict in winners_per_exp_type.items():
        df_winners[solver]['percent successful'] = pd.Series(exp_type_winner_dict)
        df_winners[solver]['SEM percent successful'] = pd.Series({e: 0 for e in exp_type_winner_dict.keys()})

    return df, df_winners


def get_dfs(df, titles, conds_plot_sep):
    dfs_to_plot = {}
    for solver, conds in conds_plot_sep.items():
        dataframe = pd.DataFrame()
        dataframe['experiment type'] = conds
        for title in titles:
            fs = {size: [(solver + ' ' + size + ' ' + c).strip() for c in cond_list]
                  for size, cond_list in conds.items()}
            filename_sets = {size: flatten([filenames[fi] for fi in f]) for size, f in fs.items()}

            values = {size: flatten(df[df['filename'].isin(filename_set)][title].tolist())
                      for size, filename_set in filename_sets.items()
                      if len(df[df['filename'].isin(filename_set)][title].tolist()) > 0}

            d = {size: np.nanmean(value) for size, value in values.items()}
            d_sem = {size: np.nanstd(value) / np.sqrt(len(value)) for size, value in values.items()}

            dataframe[title] = dataframe['experiment type'].map(d)
            dataframe['SEM ' + title] = dataframe['experiment type'].map(d_sem)
        dfs_to_plot[solver] = dataframe.copy()
    return dfs_to_plot


def plot_dfs(dfs_to_plot, titles, axs, color='blue', with_label=True):
    for (solver, dataframe), ax in zip(dfs_to_plot.items(), axs):
        ys = {size: i for i, size in enumerate(dataframe.index)}
        bar_height = 1 - 0.3 / len(titles)
        for i, t in enumerate(titles):
            if with_label:
                tick_label = [label_abbreviations[label]
                              if label in label_abbreviations.keys() else label
                              for label in dataframe.index]
            else:
                tick_label = ['' for _ in dataframe.index]
            errors = dataframe['SEM ' + t].to_frame(t)
            ax.barh(y=[ys[size] + i * bar_height for size in dataframe.index],
                    width=dataframe[t],
                    xerr=errors[t].to_list(),
                    label=t,
                    # color=color_dict[t.split(' ')[2]],
                    color=color,
                    tick_label=tick_label,
                    height=bar_height
                    )
            ax.set_ylim([-bar_height, len(dataframe)])
        if with_label:
            ax.set_ylabel(solver)
            ax.yaxis.set_label_coords(-0.7, 0.5)
    if len(titles) > 1:
        plt.legend()
    else:
        axs[-1].set_xlabel(titles[0])


if __name__ == '__main__':
    # filename = 'large_20210422152915_20210422153754'
    # ts = time_series_dict_selected_states[filename]
    #
    # t = Trajectory(filename, ts)

    ts = []
    df = pd.concat([myDataFrame.df, myDataFrame_sim.df])
    for filename, time_series, winner in \
            zip(df['filename'], df['time series'], df['winner']):
        # if filename == 'sim20220918-030902':
        #     DEBUG = 1
        trajectory = Trajectory(filename, time_series, winner)
        trajectory.find_loops()
        ts.append(trajectory)
    #
    # df_winner = df[df['winner']].copy()
    # df_winner['entered to eg'] = df_winner['filename'].map({t.filename: int('eg' in t.time_series) for t in ts})
    # df_winner['entered to eb'] = df_winner['filename'].map({t.filename: int('eb' in t.time_series) for t in ts})
    # fig, axs = plot_df(df_winner, ['entered to eg', 'entered to eb'], sharex=True, color=color_dict[t.split(' ')[2]])
    # [ax.set_xlim([0, 1]) for ax in axs]
    # save_fig(fig, 'eb_eg')

    # df = find_traps(df)
    # conds_plot_sep = cond_to_include_split_comm
    # fig, axs = plt.subplots(len(conds_plot_sep), sharex='col',
    #                         gridspec_kw={'height_ratios':
    #                                      [len(solver_dict) for solver, solver_dict in conds_plot_sep.items()]})
    # plot_df(df, ['repetition of c loop', 'repetition of b loop', 'repetition of e loop'], axs,
    #         color=color_dict[t.split(' ')[2]])

    # df = find_rotations(df)
    # conds_plot_sep = cond_to_include_split_comm
    # fig, axs = plt.subplots(len(conds_plot_sep), sharex='col',
    #                         gridspec_kw={'height_ratios':
    #                                      [len(solver_dict) for solver, solver_dict in conds_plot_sep.items()]})
    # plot_df(['repetition of ac ab loop'], axs, sharex=True, color=color_dict[t.split(' ')[2]])

    cut_off = 200
    conds_plot_sep = cond_to_include_split_comm
    df_number_of_transitions, winners = find_number_of_transitions(df.copy(), cut_off=cut_off,
                                                                   conds_plot_sep=cond_to_include_split_comm)

    fig, axs = plt.subplots(len(conds_plot_sep), 2, sharex='col',
                            gridspec_kw={'height_ratios':
                                         [len(solver_dict) for solver, solver_dict in conds_plot_sep.items()]})
    dfs = get_dfs(df_number_of_transitions, ['number of transitions'], cond_to_include_split_comm)

    plot_dfs(winners, ['percent successful'], axs[:, 0], color='blue')
    plot_dfs(dfs, ['number of transitions'], axs[:, 1], color='red', with_label=False)
    plt.subplots_adjust(hspace=.0, wspace=.1)

    save_fig(fig, 'number_of_transitions_' + str(cut_off))
