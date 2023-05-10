import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from colors import colors_humans as colors
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human
from DataFrame.plot_dataframe import save_fig
from os import path
from Directories import network_dir, home
import json
from itertools import groupby
from PS_Search_Algorithms.human_network_simulation.StateMachine import HumanStateMachine
from PS_Search_Algorithms.human_network_simulation.ManyStateMachines import ManyStateMachines


def len_in_series(x: list, first_state, last_state):
    # and is cut off at the beginning until first occurence of first_state
    # return list which is cut off after first occurence of last_state
    try:
        return int(len(x[x.index(first_state):x.index(last_state)]))
    except ValueError:
        return np.NaN


def states_series(ts):
    return [''.join(ii[0]) for ii in groupby([tuple(label) for label in ts])]


def plot_CDF(measure: pd.Series, winner: pd.Series, ax, color, label=None):
    x_values = np.arange(0, measure.max()+2, step=1)
    y_values = []

    for x in x_values:
        suc = ((measure < x) & winner).sum()
        y_values.append(suc / len(measure))
    if label is None:
        label = size
    ax.step(x_values, y_values, label=label, color=color, linewidth=2)


with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

# df_sim_small_human = HumanStateMachine.df()
# df_sim_medium_human = ManyStateMachines.df('Medium', 'states_randomChoice_8')
# df_sim_large_human = ManyStateMachines.df('Large', 'states_randomChoice_20')

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

solver = 'ant'
df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant, dfs_ant),
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]
df_solver['time_series'] = df_solver['filename'].map(time_series_dict)
perfect_states = ['ab', 'ac', 'c', 'e', 'f', 'h', 'i']

df_solver['states_series'] = df_solver['time_series'].map(lambda x: states_series(x))

# for df in [df_solver, df_sim_small_human, df_sim_medium_human, df_sim_large_human]:
#     df[measure] = df['states_series'].map(lambda x: len(len_in_series(x)))

first_state = 'ab'
last_state = 'c'
measure = 'states passed ' + first_state + '->' + last_state


for df in [df_solver]:
    first_state = 'ab'
    last_state = 'c'
    measure = 'states passed ' + first_state + '->' + last_state

    df[measure] = df['states_series'].map(lambda x: len_in_series(x, first_state, last_state))

    first_state = 'ab'
    last_state = 'h'
    measure = 'states passed ' + first_state + '->' + last_state
    df[measure] = df['states_series'].map(lambda x: len_in_series(x, first_state, last_state))

    first_state = 'c'
    last_state = 'h'
    measure = 'states passed ' + first_state + '->' + last_state
    df[measure] = df['states_series'].map(lambda x: len_in_series(x, first_state, last_state))

fig, ax = plt.subplots()
plt.xlabel(measure)
plt.ylabel('percentage reached')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]
    plot_CDF(df[measure], df['winner'], ax, colors[size])


# plot_CDF(df_sim_small_human[measure],
#          df_sim_small_human['winner'], ax,
#          colors['Small sim'], label='Small sim')

# plot_CDF(df_sim_medium_human['states passed'],
# df_sim_medium_human['winner'], ax, colors['Medium sim'], label='Medium sim')
# plot_CDF(df_sim_large_human['states passed'], df_sim_large_human['winner'], ax, colors['Large sim'], label='Large sim')


plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 15})

# draw vertical line at x = len(perfect_states)
plt.axvline(x=len_in_series(perfect_states, None, None), color='k', linestyle='--', linewidth=2)
# write 'minimal' next to this line
plt.text(len_in_series(perfect_states, None, None) + 0.5, 0.5, 'minimal', rotation=90, fontsize=15)

save_fig(fig, solver + measure)
DEBUG = 1
