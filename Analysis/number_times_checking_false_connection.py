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
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human, dfs_ant_old

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
          }

colors_solver = {'human': '#31efec',
                 'ant': '#9600ff',
                 'pheidole': '#FF8600'
                 }
colors_forks = {'ab_b_or_c': '#31efec', 'b_b1b2_or_ab': '#31ef64', 'c_cg_or_e': '#efe631', 'e_f_or_eg': '#ef8031',
                'c_ac_or_e': '#ef3131'}

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
          }

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

font = {'family': 'Times New Roman',
        # 'weight' : 'bold',
        'size': 18}
plt.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}
label_size = 12
plt.rcParams['xtick.labelsize'] = label_size

# solver = 'ant'
# categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']

solver = 'pheidole'
categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']

# solver = 'human'
# categories = ['Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']

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

        if len(state2_state) == 0 and len(state1_state) > 0:
            return True
        if len(state2_state) == 0 and len(state1_state) == 0:
            return None
        if len(state2_state) > 0 and len(state1_state) == 0:
            return False
        if state1_state[0] < state2_state[1]:
            return True
        if state1_state[0] > state2_state[1]:
            return False
        raise Exception('Hey')

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
    ab_b_or_c, b_b1b2_or_ab, c_cg_or_e, e_f_or_eg, c_ac_or_e = ({} for _ in range(5))
    for filename in filenames:
        ts = time_series_dict[filename]

        print(filename)
        # if filename == 'large_20220916093357_20220916093455':
        #     DEBUG = 1

        t = Trajectory(filename, ts)
        ab_b_or_c[filename] = {nth_time: t.state1_before_state2('b', 'c', time_series=t.after_exited('ab', nth_time))
                               for nth_time in nth_times}
        b_b1b2_or_ab[filename] = {
            nth_time: t.state1_before_state2(['b1', 'b2'], 'ab', time_series=t.after_exited('b', nth_time))
            for nth_time in nth_times}
        c_cg_or_e[filename] = {nth_time: t.state1_before_state2('cg', 'e', time_series=t.after_exited('c', nth_time))
                               for nth_time in nth_times}
        e_f_or_eg[filename] = {nth_time: t.state1_before_state2('f', 'eg', time_series=t.after_exited('e', nth_time))
                               for nth_time in nth_times}
        c_ac_or_e[filename] = {nth_time: t.state1_before_state2('ac', 'e', time_series=t.after_exited('c', nth_time))
                               for nth_time in nth_times}
    return ab_b_or_c, b_b1b2_or_ab, c_cg_or_e, e_f_or_eg, c_ac_or_e


def reduce_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 10})


def plot():
    means, std, sem = {}, {}, {}
    for fork in forks:
        for size, df_size in dfs_solver.items():
            df = df_solver[df_solver['filename'].isin(df_size['filename'])]
            means[size] = df[fork].mean()
            std[size] = df[fork].std()
            sem[size] = df[fork].std() / df[fork].count()
        ax.errorbar(list(means.keys()), list(means.values()), yerr=list(sem.values()),
                    label=fork, color=colors_forks[fork])
        ax.legend(prop={'size': 10})
        # ax.axhline(y=0, color='k', linestyle='--')
        # ax.axhline(y=1, color='k', linestyle='--')
        # ax.set_ylim([-0.1, 1.1])
    # save_fig(fig, 'rechecked_false_connection_' + solver)


def number_of_times_false_connection(source_1_2: dict, first_one_right: bool) -> dict:
    false_tests = {}
    for exp, decisions in source_1_2.items():
        dict_of_decisions = {n: boo for n, boo in decisions.items() if boo is not None}
        false_tests[exp] = (np.array(list(dict_of_decisions.values())) != first_one_right).sum()
    return false_tests


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3)
    plt.xticks(fontsize=10)
    for solver, ax in zip(['human', 'ant', 'pheidole'], axs):
        df_solver, dfs_solver = {'human': (df_human, dfs_human),
                                 'ant': (df_ant, dfs_ant),
                                 'pheidole': (df_pheidole, dfs_pheidole)}[solver]

        forks = ['ab_b_or_c', 'b_b1b2_or_ab', 'c_cg_or_e', 'e_f_or_eg', 'c_ac_or_e']
        nth_times = [i for i in range(10)]

        ab_b_or_c, b_b1b2_or_ab, c_cg_or_e, e_f_or_eg, c_ac_or_e = get_decisions_at_forks(df_solver['filename'],
                                                                                          nth_times)
        df_solver['ab_b_or_c'] = df_solver['filename'].map(number_of_times_false_connection(ab_b_or_c, False))
        df_solver['b_b1b2_or_ab'] = df_solver['filename'].map(number_of_times_false_connection(b_b1b2_or_ab, False))
        df_solver['c_cg_or_e'] = df_solver['filename'].map(number_of_times_false_connection(c_cg_or_e, False))
        df_solver['e_f_or_eg'] = df_solver['filename'].map(number_of_times_false_connection(e_f_or_eg, True))
        df_solver['c_ac_or_e'] = df_solver['filename'].map(number_of_times_false_connection(c_ac_or_e, False))
        plot()

        ax.set_title(solver)
        ax.set_xlabel('size')

    fig.suptitle('Number of times the false connection was made')
    fig.set_size_inches(15, 7)
    save_fig(fig, 'rechecked_false_connection')
    DEBUG = 1

    #
    # forks = ['c_ac_or_e']
    # nth_times = [i for i in range(6)]
    # c_ac_or_e = get_decisions_at_c(df_solver['filename'], nth_times)
    # df_solver['c_ac_or_e'] = df_solver['filename'].map(c_ac_or_e)
    # plot_nth_decision(n=0)
