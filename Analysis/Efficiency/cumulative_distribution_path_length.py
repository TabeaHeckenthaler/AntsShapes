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

with open(translation_dir, 'r') as json_file:
    translation_dict = json.load(json_file)
    json_file.close()

with open(rotation_dir, 'r') as json_file:
    rotation_dict = json.load(json_file)
    json_file.close()

with open(minimal_filename_dict, 'r') as json_file:
    minimal_filename_dict = json.load(json_file)
    json_file.close()

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

solver = 'human'

df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant, dfs_ant),
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]

df_solver['minimal filename'] = df_solver['filename'].map(minimal_filename_dict)
df_solver['translation'] = df_solver['filename'].map(translation_dict)
df_solver['rotation'] = df_solver['filename'].map(rotation_dict)
df_solver['minimal translation'] = df_solver['minimal filename'].map(translation_dict)
df_solver['minimal rotation'] = df_solver['minimal filename'].map(rotation_dict)
df_solver['norm translation'] = df_solver['translation'] / df_solver['minimal translation']
df_solver['norm rotation'] = df_solver['rotation'] / df_solver['minimal rotation']

# __________ROT  +  TRANS_____________

fig, ax = plt.subplots()
plt.xlabel('normalized translation + rotation')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]

    df['path length'] = df['norm translation'] + df['norm rotation']

    x_values = np.arange(0, df['path length'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['path length'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
save_fig(fig, solver + 'cum_distribution_path_length')
DEBUG = 1

# _______________________



fig, ax = plt.subplots()
plt.xlabel('normalized translation')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]

    x_values = np.arange(0, df['norm translation'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm translation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
save_fig(fig, solver + 'cum_distribution_translation')
DEBUG = 1


fig, ax = plt.subplots()
plt.xlabel('normalized rotation')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]

    x_values = np.arange(0, df['norm rotation'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm rotation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
save_fig(fig, solver + 'cum_distribution_rotation')
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
