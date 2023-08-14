# adapted from Analysis.PathPy.Loops
import warnings
from DataFrame.plot_dataframe import save_fig
from DataFrame.Altered_DataFrame import myDataFrame
from matplotlib import pyplot as plt
from typing import Union
from itertools import groupby
import numpy as np
from Directories import averageCarrierNumber_dir
import json
import os
from Directories import network_dir
from DataFrame.import_excel_dfs import df_ant_excluded, df_pheidole, dfs_pheidole, df_human, dfs_human
from trajectory_inheritance.get import get
from DataFrame.gillespie_dataFrame import df_gillespie, dfs_gillespie, time_series_sim_dict

groups = df_ant_excluded.groupby('size')
dfs_ant_excluded = {k: v for k, v in groups}

warnings.simplefilter(action='ignore', category=FutureWarning)

label_abbreviations = {'S (> 1)': 'S (> 1 ants)',
                       'Single (1)': 'S (1 ant)',
                       'Large communication': 'Large C',
                       'M (>7) communication': 'Medium C',
                       'Large non_communication': 'Large NC',
                       'M (>7) non_communication': 'Medium NC',
                       'Small non_communication': 'Small',
                       }

# colors = {'Large C': '#8931EF',
#           'Large NC': '#FF00BD',
#           'Medium C': '#FF8600',
#           'Medium NC': '#18FF00',
#           'Small': '#000000',
#           'XL': '#ff00c1',
#           'L': '#9600ff',
#           'M': '#4900ff',
#           'S (> 1)': '#00b8ff',
#           'Single (1)': '#00fff9',
#           }

colors = {'Large C': '#8931EF',
          'Large NC': '#8931EF',
          'Medium C': '#FF8600',
          'Medium NC': '#FF8600',
          'Small': '#000000',
          'XL': '#ff00c1',
          'L': '#9600ff',
          'M': '#4900ff',
          'S (> 1)': '#00b8ff',
          'Single (1)': '#00fff9',
          'S': '#00b8ff',
          'XS': '#00fff9',
          }

markersize = {'Large C': 7,
              'Large NC': 7,
              'Medium C': 7,
              'Medium NC': 7,
              'Small': 5,
              'XL': 9,
              'L': 8,
              'M': 7,
              'S (> 1)': 6,
              'Single (1)': 5,
              'S': 6,
              'XS': 5,
              }

marker = {'Large C': 'o',
          'Large NC': 'x',
          'Medium C': 'o',
          'Medium NC': 'x',
          'Small': 'o',
          'XL': 'x',
          'L': 'x',
          'M': 'x',
          'S (> 1)': 'x',
          'Single (1)': 'x',
          'S': 'x',
          'XS': 'x',
          }

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

time_series_dict.update(time_series_sim_dict)

font = {'family': 'Times New Roman',
        # 'weight' : 'bold',
        'size': 17}
plt.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}

# solver = 'ant'
# categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
#
# solver = 'pheidole'
# categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']

solver = 'human'
categories = ['Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']

with open(averageCarrierNumber_dir, 'r') as json_file:
    averageCarrierNumber_dict = json.load(json_file)
myDataFrame['average Carrier Number'] = myDataFrame['filename'].map(averageCarrierNumber_dict)


class Trajectory:
    def __init__(self, filename, time_series, winner=None):
        self.filename = filename
        self.time_series = time_series
        self.state_series = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        self.loops = None
        self.winner = winner

    def state1_before_state2(self, state1, state2, time_series: list) -> Union[bool, None]:
        if type(state1) is list and len(state1) == 2:
            return self.state1_before_state2(state1[0], state2, time_series) or \
                self.state1_before_state2(state1[1], state2, time_series)

        state1_state = np.where(np.array(time_series) == state1)[0]
        state2_state = np.where(np.array(time_series) == state2)[0]

        # if state2 was never reached, but state1 was
        if len(state2_state) == 0 and len(state1_state) > 0:
            return True

        # if state2 and state1 was never reached
        if len(state2_state) == 0 and len(state1_state) == 0:
            return None

        # if state1 was never reached, but state2 was
        if len(state2_state) > 0 and len(state1_state) == 0:
            return False

        if state1_state[0] < state2_state[0]:
            return True

        if state1_state[0] > state2_state[0]:
            return False
        raise Exception('We never reached any case')

    def after_exited(self, state: Union[str, list], nth: int) -> list:
        """
        Index when state was exited for the nth time.
        """
        if type(state) == str:
            state = [state]
        in_state = np.isin(np.array(['0'] + self.time_series + ['0']), state).astype(int)
        entered_exited = np.where(in_state[:-1] != in_state[1:])[0]

        if entered_exited.size == 0:
            return []

        times = np.split(entered_exited, int(len(entered_exited) / 2))
        # time_series[0:15] = 'ab1'
        if len(times) <= nth:
            return []
        if len(times[nth]) < 2:
            raise Exception()
        return self.time_series[times[nth][1]:]


def get_decisions_at_forks(filenames: list, nth_times: list) -> tuple:
    ab_b_or_c, b_b1b2_or_be, c_cg_or_e, e_f_or_eg, c_ac_or_e = ({} for _ in range(5))
    for filename in filenames:
        ts = time_series_dict[filename]

        print(filename)
        # if filename == 'large_20220916093357_20220916093455':
        #     DEBUG = 1

        t = Trajectory(filename, ts)
        ab_b_or_c[filename] = {nth_time: t.state1_before_state2('b', 'c', time_series=t.after_exited('ab', nth_time))
                               for nth_time in nth_times}
        b_b1b2_or_be[filename] = {
            nth_time: t.state1_before_state2(['b1', 'b2'], 'be', time_series=t.after_exited('b', nth_time))
            for nth_time in nth_times}
        c_cg_or_e[filename] = {nth_time: t.state1_before_state2('cg', 'e', time_series=t.after_exited('c', nth_time))
                               for nth_time in nth_times}
        e_f_or_eg[filename] = {nth_time: t.state1_before_state2('f', 'eg', time_series=t.after_exited('e', nth_time))
                               for nth_time in nth_times}
        c_ac_or_e[filename] = {nth_time: t.state1_before_state2('ac', 'e', time_series=t.after_exited('c', nth_time))
                               for nth_time in nth_times}
    return ab_b_or_c, b_b1b2_or_be, c_cg_or_e, e_f_or_eg, c_ac_or_e


def get_decisions_at_c(filenames: list, nth_times: list) -> dict:
    c_ac_or_e = {}
    for filename in filenames:
        ts = time_series_dict[filename]

        print(filename)
        # if filename == 'large_20220916093357_20220916093455':
        #     DEBUG = 1
        t = Trajectory(filename, ts)
        c_ac_or_e[filename] = {nth_time: t.state1_before_state2('ac', 'e', time_series=t.after_exited('c', nth_time))
                               for nth_time in nth_times}
    return c_ac_or_e


def reduce_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # move to area, where it does not overlap with the plot
    ax.legend(by_label.values(), by_label.keys(), prop={'size': 15}, bbox_to_anchor=(0, 0), loc='lower left')


def plot_1st_decision():
    """
    Plot the first decision for each fork.
    :return:
    """


def plot_nth_decision(n: int = 0, ax=None):
    for fork, x_pos in zip(forks, range(len(forks))):
        shift = -0.1
        for size, df_size in dfs_solver.items():
            df = df_solver[df_solver['filename'].isin(df_size['filename'])]
            c_withNones = {n: np.array([d[str(n)] for d in df[fork]]) for n in nth_times}
            c_withoutNones = {n: c[c != np.array(None)] for n, c in c_withNones.items()}

            results = {key: decisions.sum() / len(decisions) if not len(decisions) == 0 else None
                       for key, decisions in c_withoutNones.items()}
            error = {}
            for key, decisions in c_withoutNones.items():
                if len(decisions) > 8 and results[key] is not None:
                    # https://en.wikipedia.org/wiki/Binomial_distribution
                    error[key] = np.sqrt(results[key] * (1 - results[key]) / (len(decisions) - 1))
                else:
                    error[key] = 1

            if results[n] is not None:
                ax.errorbar(x_pos + shift, results[n], yerr=error[n], label=size, color=colors[size],
                            marker=marker[size], markersize=markersize[size])
            shift += 0.05

    ax.set_xticks(range(len(forks)))
    ax.set_xticklabels(forks)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axhline(y=1, color='k', linestyle='--')
    ax.set_ylim([-0.1, 1.1])
    ax.set_title(solver, fontsize=17)
    # increase font size of legend
    reduce_legend(ax)
    # place legend at bottom left
    # ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, prop={'size': 17})


def plot_nth_decisions(fig, axs):
    axs = axs.flatten()
    for fork, ax in zip(forks, axs):
        print(fork)
        for size, df_size in dfs_solver.items():
            df = df_solver[df_solver['filename'].isin(df_size['filename'])]
            c_withNones = {n: np.array([d[str(n)] for d in df[fork]]) for n in nth_times}
            c_withoutNones = {n: c[c != np.array(None)] for n, c in c_withNones.items()}

            # find last value which is not None in df_to_last_decision.iloc[0]
            def funct(df_copy):
                last_index = [k for k, v in df_copy.items() if v is not None]
                if len(last_index) == 0:
                    return None
                fixed = {k: v if v is not None else df_copy[last_index[-1]] for k, v in df_copy.items()}
                return fixed

            df_to_last_decision = df[fork].map(funct).dropna()
            c_Nones_to_last = {n: np.array([d[str(n)] for d in df_to_last_decision]) for n in nth_times}

            c = c_withoutNones
            # c = c_Nones_to_last

            results = {key: decisions.sum() / len(decisions)
            if len(decisions) > 0 else np.NaN
                       for key, decisions in c.items()}

            error = {}
            for key, decisions in c.items():
                if len(decisions) > 1 and not np.isnan(results[key]):
                    # https://en.wikipedia.org/wiki/Binomial_distribution
                    error[key] = np.sqrt(results[key] * (1 - results[key])) / (len(decisions) - 1)
                else:
                    error[key] = 1

            if not np.isnan(results[0]):
                ax.errorbar(list(results.keys()), list(results.values()), yerr=list(error.values()), label=size,
                            color=colors[size], marker=marker[size], markersize=markersize[size])

        ax.legend(prop={'size': 10})
        ax.set_title(fork)
        ax.axhline(y=0, color='k', linestyle='--')
        ax.axhline(y=1, color='k', linestyle='--')
        ax.set_ylim([-0.1, 1.1])

        # make xticklabels integers
        ax.set_xticks(range(0, 20, 2))
        ax.set_xticklabels(range(0, 20, 2))

        # set xlabel and ylabel
        ax.set_xlabel('visit number')
        ax.set_ylabel('decision')

    fig.tight_layout()
    fig.set_size_inches(5, 7)
    save_fig(fig, 'long_memory_less' + '_' + solver)


def plot_0th_decision_all_solvers():
    fig, axs = plt.subplots(1, 3, figsize=(22, 5))
    for ax, solver in zip(axs, ['human', 'ant', 'gillespie', ]):
        print(solver)
        forks = ['ab_b_or_c', 'b_b1b2_or_be', 'c_cg_or_e', 'e_f_or_eg']
        df_solver, dfs_solver = {'human': (df_human, dfs_human),
                                 'ant': (df_ant_excluded, dfs_ant_excluded),
                                 'pheidole': (df_pheidole, dfs_pheidole),
                                 'gillespie': (df_gillespie, dfs_gillespie)}[solver]
        nth_times = [i for i in range(20)]

        # # save decisions at forks as json
        # ab_b_or_c, b_b1b2_or_be, c_cg_or_e, e_f_or_eg, c_ac_or_e = get_decisions_at_forks(df_solver['filename'],
        #                                                                                   nth_times)
        # with open(solver + '_decisions_at_forks.json', 'w') as f:
        #     json.dump({'ab_b_or_c': ab_b_or_c, 'b_b1b2_or_be': b_b1b2_or_be, 'c_cg_or_e': c_cg_or_e,
        #                'e_f_or_eg': e_f_or_eg, 'c_ac_or_e': c_ac_or_e}, f)

        with open(solver + '_decisions_at_forks.json', 'r') as f:
            d = json.load(f)

        len({k for k, di in d['e_f_or_eg'].items() if di['0'] and '_XS_' in k})

        df_solver['ab_b_or_c'] = df_solver['filename'].map(d['ab_b_or_c'])
        df_solver['b_b1b2_or_be'] = df_solver['filename'].map(d['b_b1b2_or_be'])
        df_solver['c_cg_or_e'] = df_solver['filename'].map(d['c_cg_or_e'])
        df_solver['e_f_or_eg'] = df_solver['filename'].map(d['e_f_or_eg'])
        df_solver['c_ac_or_e'] = df_solver['filename'].map(d['c_ac_or_e'])
        plot_nth_decision(n=0, ax=ax)
    save_fig(fig, '0th_decision_all_solvers')


def mistakes(decisions, invert=False):
    if invert:
        decisions = [not d for d in decisions.copy()]
    return np.array([dec for dec in list(decisions.values()) if dec is not None]).sum()


if __name__ == '__main__':

    fig, axs = plt.subplots(2, 3)
    for ax, solver in zip(axs, ['human', 'gillespie', 'ant', ]):
        print(solver)
        # forks = ['ab_b_or_c', 'b_b1b2_or_be', 'c_cg_or_e', 'e_f_or_eg']
        # forks = ['ab_b_or_c', 'b_b1b2_or_be', 'c_cg_or_e', 'e_f_or_eg', 'c_ac_or_e']
        forks = ['ab_b_or_c', 'c_cg_or_e']
        # fig, axs = plt.subplots(2, 3)
        df_solver, dfs_solver = {'human': (df_human, dfs_human),
                                 'ant': (df_ant_excluded, dfs_ant_excluded),
                                 'pheidole': (df_pheidole, dfs_pheidole),
                                 'gillespie': (df_gillespie, dfs_gillespie)}[solver]
        nth_times = [i for i in range(20)]

        # # save decisions at forks as json
        # ab_b_or_c, b_b1b2_or_be, c_cg_or_e, e_f_or_eg, c_ac_or_e = get_decisions_at_forks(df_solver['filename'],
        #                                                                                   nth_times)
        # with open(solver + '_decisions_at_forks.json', 'w') as f:
        #     json.dump({'ab_b_or_c': ab_b_or_c, 'b_b1b2_or_be': b_b1b2_or_be, 'c_cg_or_e': c_cg_or_e,
        #                'e_f_or_eg': e_f_or_eg, 'c_ac_or_e': c_ac_or_e}, f)

        with open(solver + '_decisions_at_forks.json', 'r') as f:
            d = json.load(f)
        df_solver['ab_b_or_c'] = df_solver['filename'].map(d['ab_b_or_c'])
        # df_solver['b_b1b2_or_be'] = df_solver['filename'].map(d['b_b1b2_or_be'])
        df_solver['c_cg_or_e'] = df_solver['filename'].map(d['c_cg_or_e'])
        # df_solver['e_f_or_eg'] = df_solver['filename'].map(d['e_f_or_eg'])
        # df_solver['c_ac_or_e'] = df_solver['filename'].map(d['c_ac_or_e'])

        # find all rows where df_solver['c_cg_or_e']['4'] is False
        outlier3 = df_solver[df_solver['c_cg_or_e'].apply(lambda x: x['3'] is not None)][['filename', 'c_cg_or_e', 'communication', 'size']]
        outlier4 = df_solver[df_solver['c_cg_or_e'].apply(lambda x: x['4'] is not None)][['filename', 'c_cg_or_e', 'communication', 'size']]

        for i, row in outlier4[['filename', 'communication']].iterrows():
            filename, com = row['filename'], row['communication']
            labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series_dict[filename]])]
            print(filename, ', communication: ', com, '\n', labels, '\n')

        fig, axs1 = plt.subplots(2)
        plot_nth_decisions(fig, axs1)
        DEBUG = 1

    #     for fork_name, (fork, ax) in {'ab_b_or_c': (ab_b_or_c, axs[0]), 'c_ac_or_e': (c_ac_or_e, axs[1])}.items():
    #         mean_number_mistakes, sem_number_mistakes = {}, {}
    #         # find mean number of entries to b from ac
    #         for group_type, df in dfs_solver.items():
    #             decisions = df['filename'].map(fork)
    #             # if df['filename'] == 'la rge_20220916093357_20220916093455':
    #             #     DEBUG = 1
    #             df['mistakes'] = decisions.apply(mistakes)
    #             mean_number_mistakes[group_type] = np.mean(df['mistakes'])
    #             sem_number_mistakes[group_type] = np.std(df['mistakes'])/len(df['mistakes'])
    #
    #         # plot mean number of mistakes
    #         ax.errorbar(list(mean_number_mistakes.keys()), list(mean_number_mistakes.values()),
    #                             yerr=list(sem_number_mistakes.values()))
    #         ax.set_title('Mean number of mistakes: ' + fork_name)
    #         # plot a horizontal line at 1
    #         ax.axhline(y=1, color='k', linestyle='--')
    #         # set x_axis range to 0 to maximal value
    #         ax.set_ylim([0, max(mean_number_mistakes.values()) + max(sem_number_mistakes.values())])
    # # prevent axis labels from overlapping
    # fig.tight_layout()
    # save_fig(fig, 'mean_number_mistakes')
    # plot_nth_decisions(fig, axs1)
    # DEBUG = 1

    #
    # forks = ['c_ac_or_e']
    # nth_times = [i for i in range(6)]
    # c_ac_or_e = get_decisions_at_c(df_solver['filename'], nth_times)
    # df_solver['c_ac_or_e'] = df_solver['filename'].map(c_ac_or_e)
