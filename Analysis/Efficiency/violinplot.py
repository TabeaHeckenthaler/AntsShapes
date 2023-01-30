import matplotlib.pyplot as plt
from trajectory_inheritance.exp_types import solver_geometry
from DataFrame.plot_dataframe import save_fig
import numpy as np
from trajectory_inheritance.exp_types import color
from Analysis.average_carrier_number.averageCarrierNumber import averageCarrierNumber_dict
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human
import json
from Directories import home
from os import path

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})

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

solver = 'pheidole'

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

trans_dict = {}

for size, df_size in dfs_solver.items():
    trans_dict[size] = df_solver[df_solver['filename'].isin(df_size['filename'])]['norm translation'].values

dataset = [v for v in trans_dict.values()]
# positions = [0.2, 0.8, 2.2, 2.8, 4.2]
positions = [k for k in range(len(dataset))]

fig, axes = plt.subplots()
plt.xlabel('probability density distribution')
plt.ylabel('normalized path length')
plt.title(solver + ' PDF')
parts = axes.violinplot(dataset=dataset, positions=positions, showmeans=1, showextrema=0)
# axes.set_xticks(tick_positions)
# axes.set_xticklabels(sizes[::2])
#
# for size, position in zip(sizes[::2], tick_positions):
#     succ = len(dfss[size]['winner']['norm path length'].values)/\
#            (len(dfss[size]['looser']['norm path length'].values) + len(dfss[size]['winner']['norm path length'].values))
#     plt.text(position-0.8, 60, str(round(succ, 2))[:4])
#
# for pc, label in zip(parts['bodies'][::2], sizes[::2]):
#     pc.set_facecolor(color[label])
#     pc.set_alpha(1)
# for pc, label in zip(parts['bodies'][1::2], sizes[1::2]):
#     pc.set_facecolor('#ffc2c2')
#     pc.set_alpha(1)
#     pc.set_edgecolor('black')
# parts['cmeans'].set_color('black')
# fig.set_size_inches([8.35, 6.63])
# save_fig(fig, solver + '_cum_distribution')
# DEBUG = 1


# solver = 'human'
# adf.choose_experiments(solver=solver, shape='SPT', geometry=solver_geometry[solver], init_cond='back')
# adf.df['average Carrier Number'] = adf.df['filename'].map(averageCarrierNumber_dict)
#
# dfss = adf.get_separate_data_frames(solver, plot_separately[solver])
#
# del dfss['M (1)']
# del dfss['M (2)']
# del dfss['Small']['communication']
#
# sizes = ['Large', 'Large', 'M (>7)', 'M (>7)', 'Small']
# comms = ['communication', 'non_communication', 'communication', 'non_communication', 'non_communication']
# dataset = [dfss[size][comm]['norm path length'].values for size, comm in zip(sizes, comms)]

# positions = [0.2, 0.8, 2.2, 2.8, 4.2]
# tick_positions = [0.5, 2.5, 4.2]
#
# fig, axes = plt.subplots()
# plt.xlabel('probability density distribution')
# plt.ylabel('normalized path length')
# plt.title(solver + ' PDF')
# parts = axes.violinplot(dataset=dataset, positions=positions, showmeans=1, showextrema=0)
#
# means = [np.mean(d) for d in dataset]
# sems = [np.std(d)/np.sqrt(len(d)) for d in dataset]
# axes.errorbar(positions, means, yerr=sems, linestyle='', color='k', capsize=10)
#
#
# axes.set_xticks(tick_positions)
# axes.set_xticklabels(sizes[::2])
#
# for position_nc, position_c in zip(positions[1::2], positions[::2]):
#     plt.text(position_nc-0.1, 6, 'NC')
#     plt.text(position_c-0.1, 6, 'C')
#
# for pc, size, comm in zip(parts['bodies'], sizes, comms):
#     pc.set_facecolor(color[size + ' ' + comm])
#     pc.set_alpha(1)
#     # pc.set_edgecolor('black')
# parts['cmeans'].set_color('black')
# fig.set_size_inches([8.35, 6.63])
# save_fig(fig, solver + '_cum_distribution')


# _______________ Grouped together Large and Medium while separating C and NC
#
# sizes = ['Large', 'Large', 'M (>7)', 'M (>7)', 'Small']
# comms = ['communication', 'non_communication', 'communication', 'non_communication', 'non_communication']
# positions = [0.2, 2.2, 4.2]
# tick_positions = [0.5, 2.5, 4.2]
#
# dfss_grouped = {}
# dfss_grouped['C group'] = dfss['Large']['communication']['norm path length'].values.tolist() + \
#                                       dfss['M (>7)']['communication']['norm path length'].values.tolist()
# dfss_grouped['NC group'] = dfss['Large']['non_communication']['norm path length'].values.tolist() + \
#                                       dfss['M (>7)']['non_communication']['norm path length'].values.tolist()
# dfss_grouped['single'] = dfss['Small']['non_communication']['norm path length'].values.tolist()
#
# dataset = [values for keys, values in dfss_grouped.items()]
#
# fig, axes = plt.subplots()
# plt.xlabel('probability density distribution')
# plt.ylabel('normalized path length')
# plt.title(solver + ' PDF')
# parts = axes.violinplot(dataset=dataset, positions=positions, showmeans=1, showextrema=0)
#
# means = [np.mean(d) for d in dataset]
# sems = [np.std(d)/np.sqrt(len(d)) for d in dataset]
# axes.errorbar(positions, means, yerr=sems, linestyle='', color='k', capsize=10)
#
# axes.set_xticks(positions)
# axes.set_xticklabels(dfss_grouped.keys())
#
# for position_nc, position_c in zip(positions[1::2], positions[::2]):
#     plt.text(position_nc-0.1, 6, 'NC')
#     plt.text(position_c-0.1, 6, 'C')
#
# for pc, size, comm in zip(parts['bodies'], sizes, comms):
#     pc.set_facecolor(color[size + ' ' + comm])
#     pc.set_alpha(1)
#     # pc.set_edgecolor('black')
# parts['cmeans'].set_color('black')
# fig.set_size_inches([8.35, 6.63])
# save_fig(fig, solver + '_cum_distribution_grouped')

DEBUG = 1
