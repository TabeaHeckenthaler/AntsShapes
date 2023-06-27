import numpy as np
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import df_ant_excluded, df_pheidole, dfs_pheidole, df_human, dfs_human
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

solver = 'ant'

df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant_excluded, None), # still have to do this
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]

df_solver['minimal filename'] = df_solver['filename'].map(minimal_filename_dict)
df_solver['translation'] = df_solver['filename'].map(translation_dict)
df_solver['rotation'] = df_solver['filename'].map(rotation_dict)
df_solver['minimal translation'] = df_solver['minimal filename'].map(translation_dict)
df_solver['minimal rotation'] = df_solver['minimal filename'].map(rotation_dict)
df_solver['norm translation'] = df_solver['translation'] / df_solver['minimal translation']
df_solver['norm rotation'] = df_solver['rotation'] / df_solver['minimal rotation']

# based on getting stuck in b
# exclude = ['L_SPT_4420007_LSpecialT_1_ants (part 1)',
#            'L_SPT_5030006_LSpecialT_1_ants (part 1)',
#            'L_SPT_4420005_LSpecialT_1_ants (part 1)',
#            'L_SPT_4670001_LSpecialT_1_ants (part 1)',
#            'L_SPT_4420010_LSpecialT_1_ants (part 1)',
#            'M_SPT_4700003_MSpecialT_1_ants (part 1)',
#            'L_SPT_4670006_LSpecialT_1_ants (part 1)',
#            'L_SPT_5030001_LSpecialT_1_ants (part 1)',
#            'S_SPT_4760017_SSpecialT_1_ants (part 1)',
#            'L_SPT_5010001_LSpecialT_1_ants (part 1)',
#            'L_SPT_4650012_LSpecialT_1_ants (part 1)',
#            'L_SPT_5030009_LSpecialT_1_ants (part 1)',
#            'L_SPT_4660006_LSpecialT_1_ants (part 1)',
#            'S_SPT_5180011_SSpecialT_1_ants (part 1)',
#            'XL_SPT_4630015_XLSpecialT_1_ants (part 1)']

sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']

# ___ MEANS ___
fig, ax = plt.subplots()
plt.xlabel('size')
plt.ylabel('normalized translation + rotation')
df_size = df_solver.groupby('size')

means, stds = {}, {}
# plot the means of the normalized translation + rotation
for size in sizes[1:]:
    df = df_solver[df_solver['size'] == size]
    df['path length'] = df['norm translation'] + df['norm rotation']
    df.loc[~df['winner'], 'path length'] = df.loc[~df['winner'], 'path length'] * 2
    # df = df[df['winner']]
    means[size] = np.mean(df['path length'])
    stds[size] = np.std(df['path length']) / np.sqrt(len(df))

plt.errorbar(means.keys(), means.values(), yerr=stds.values(), alpha=0.5)
plt.tight_layout()
plt.savefig('mean_pL' + solver + '.png')
DEBUG = 1

# __________ TIME _____________

fig, ax = plt.subplots()
plt.xlabel('normalized time')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

scale = {'Single (1)': 8, 'S (> 1)': 8, 'M': 4, 'L': 2, 'XL': 1}

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]

    df['norm time'] = df['time [s]'] * df['size'].map(scale)

    x_values = np.arange(0, df['norm time'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm time'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
plt.savefig('normalized_time_' + solver + '.png')
DEBUG = 1

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
