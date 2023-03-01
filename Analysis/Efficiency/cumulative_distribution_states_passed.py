import numpy as np
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human
from DataFrame.plot_dataframe import save_fig
from os import path
from Directories import network_dir
import json
from itertools import groupby

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

colors = {'Large C': '#8931EF',
          'Large NC': '#cfa9fc',
          'Medium C': '#FF8600',
          'Medium NC': '#fab76e',
          'Small': '#000000',
          'XL': '#ff00c1',
          'L': '#9600ff',
          'M': '#4900ff',
          'S (> 1)': '#00b8ff',
          'Single (1)': '#00fff9',
          }

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

solver = 'human'

df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant, dfs_ant),
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]

df_solver['time_series'] = df_solver['filename'].map(time_series_dict)
perfect_states = ['ab', 'ac', 'c', 'e', 'f', 'h', 'i']

def states_series(ts):
    return [''.join(ii[0]) for ii in groupby([tuple(label) for label in ts])]


df_solver['states_series'] = df_solver['time_series'].map(lambda x: states_series(x))
df_solver['states passed'] = df_solver['states_series'].map(lambda x: len(x))

fig, ax = plt.subplots()
plt.xlabel('number of states passed')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]

    x_values = np.arange(0, df['states passed'].max(), step=1)
    y_values = []

    for x in x_values:
        suc = df[(df['states passed'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 15})

# draw vertical line at x = len(perfect_states)
plt.axvline(x=len(perfect_states), color='k', linestyle='--', linewidth=2)
# write 'minimal' next to this line
plt.text(len(perfect_states) + 0.5, 0.5, 'minimal', rotation=90, fontsize=15)


save_fig(fig, solver + 'cum_distribution_states_passed')
DEBUG = 1
