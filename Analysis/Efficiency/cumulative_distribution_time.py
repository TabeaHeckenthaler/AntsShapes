import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.exp_types import solver_geometry
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.exp_types import color

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}
solver_step = {'ant': 100, 'human': 30}
size_norm = {'XL': 1, 'L': 2, 'M': 4, 'S (> 1)': 8, 'Single (1)': 8,
             'Large': 1, 'M (>7)': 2, 'M (2)': 2, 'M (1)': 2, 'Small': 4}
adf = Altered_DataFrame()

fig, ax = plt.subplots()
x_value = 'norm_time'  # 'norm_time', 'time [s]'
plt.xlabel(x_value + ' [min]')
plt.ylabel('percentage solved')
plt.show(block=0)

# solver = 'ant'
# adf.choose_experiments(solver=solver, shape='SPT', geometry=solver_geometry[solver], init_cond='back')
# dfss = adf.get_separate_data_frames(solver, plot_separately[solver])
#
# for size, dfs in dfss.items():
#     df = pd.concat(dfs, ignore_index=True)
#     df['norm_time'] = df['time [s]'] * size_norm[size]
#
#     x_values = np.arange(0, df[x_value].max(), step=solver_step[solver])
#     y_values = []
#     for x in x_values:
#         suc = df[(df[x_value] < x) & (df['winner'])]
#         y_values.append(len(suc) / len(df))
#
#     plt.step(x_values/60, y_values, label=size, color=color[size], linewidth=4,)
#
# plt.title(solver + ' CDF')
# plt.legend(prop={'size': 20})
# fig.set_size_inches([12, 6.63])
# # plt.xlabel(x_value[:-4] + ' [min]')
#
# save_fig(fig, x_value + solver + '_cum_distribution_time')
# DEBUG = 1


solver = 'human'
adf.choose_experiments(solver=solver, shape='SPT', geometry=solver_geometry[solver], init_cond='back')
dfss = adf.get_separate_data_frames(solver, plot_separately[solver])

if solver == 'human':
    del dfss['M (1)']
    del dfss['M (2)']
    del dfss['Small']['communication']

for size, dfs in dfss.items():
    for c, df in dfs.items():
        print(size, c)
        df['norm_time'] = df['time [s]'] * size_norm[size]

        x_values = np.arange(0, df[x_value].max(), step=solver_step[solver])
        y_values = []
        for x in x_values:
            suc = df[(df[x_value] < x) & (df['winner'])]
            y_values.append(len(suc) / len(df))

        plt.step(x_values/60, y_values, label=size + ' ' + c, color=color[size + ' ' + c], linewidth=4,)

plt.title(solver + ' CDF')
plt.legend(prop={'size': 20})
fig.set_size_inches([12, 6.63])
# plt.xlabel(x_value[:-4] + ' [min]')

save_fig(fig, x_value + solver + '_cum_distribution_time')
DEBUG = 1