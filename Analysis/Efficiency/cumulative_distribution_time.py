import numpy as np
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human
from DataFrame.plot_dataframe import save_fig
from os import path
from Directories import home
import json
from colors import colors_humans as colors

translation_dir = path.join(home, 'Analysis', 'Efficiency', 'translation.json')
rotation_dir = path.join(home, 'Analysis', 'Efficiency', 'rotation.json')
minimal_filename_dict = path.join(home, 'Analysis', 'minimal_path_length', 'minimal_filename_dict.json')

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

solver = 'pheidole'
df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant, dfs_ant),
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]

fig, ax = plt.subplots()
plt.xlabel('norm time [s]')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

scale_dict = {'Single (1)': 1, 'S (> 1)': 1, 'M': 2, 'L': 4, 'XL': 8,
              'Large C': 4, 'Large NC': 4, 'Medium C': 2, 'Medium NC': 2, 'Small': 1}

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]
    df['norm time [s]'] = df['time [s]'] / scale_dict[size]
    x_values = np.arange(0, df['norm time [s]'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = ((df['norm time [s]'] < x) & df['winner']).sum()
        y_values.append(suc / len(df['norm time [s]']))
    ax.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
plt.tight_layout()
save_fig(fig, solver + 'cum_distribution_time')
DEBUG = 1


# SPLIT MEN AND WOMEN
# from trajectory_inheritance.humans import give_gender
# size = 'Small'
# dfs = dfss['Small']
#
# df_all = pd.concat(dfs, ignore_index=True)
# gender_dict = give_gender(df_all['filename'])
#
# men = [f for f in gender_dict.keys() if gender_dict[f] == 'M']
# women = [f for f in gender_dict.keys() if gender_dict[f] == 'F']
#
# df_men = df_all[df_all['filename'].isin(men)]
# df_women = df_all[df_all['filename'].isin(women)]
#
#
# for gender, df in zip(['men', 'women'], [df_men, df_women]):
#     print(gender)
#     df['path length'] = df['filename'].map(path_length_dict)
#     df['mininmal path length'] = df['filename'].map(minimal_path_length_dict)
#     df['norm path length'] = df['path length'] / df['mininmal path length']
#
#     print(df['norm path length'].mean())
#     print(df['norm path length'].sem())
#
#     N = len(df)
#     x = list(df['norm path length'].sort_values())
#     y = np.arange(N) / float(N)
#
#     plt.step(x, y, label=gender)
#
# plt.legend(prop={'size': 20})
# save_fig(fig, solver + '_cum_distribution_gender')

# ALL SIZES
# for size, dfs in dfss.items():
#     if size in ['Large', 'M (>7)', 'Small']:
#         for comm, df in dfs.items():
#             df['path length'] = df['filename'].map(path_length_dict)
#             df['mininmal path length'] = df['filename'].map(minimal_path_length_dict)
#             df['norm path length'] = df['path length'] / df['mininmal path length']
#
#             cut_off = df[df['winner']]['norm path length'].max()
#             to_short = df['norm path length'] < cut_off
#             looser = ~ df['winner']
#             df.drop(df[to_short & looser].index, inplace=True)
#             df = df.reindex()
#             df.loc[~ df['winner'], 'norm path length'] = np.nan
#
#             N = len(df)
#             x = list(df['norm path length'].sort_values())
#             y = np.arange(N) / float(N)
#
#             plt.step(x, y, label=size + ' ' + comm[:5])
#
# plt.legend(prop={'size': 15})
# save_fig(fig, solver + '_cum_distribution')
# DEBUG = 1
