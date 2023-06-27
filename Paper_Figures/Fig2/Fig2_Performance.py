from matplotlib import pyplot as plt
import numpy as np
from DataFrame.import_excel_dfs import df_ant_excluded, dfs_human
from colors import colors_humans as colors
import json
from Directories import home
from os import path
import pandas as pd
from trajectory_inheritance.get import get

plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

translation_dir = path.join(home, 'Analysis', 'Efficiency', 'translation.json')
translation_dir_sim = path.join(home, 'Analysis', 'Efficiency', 'translation_sim.json')
rotation_dir = path.join(home, 'Analysis', 'Efficiency', 'rotation.json')
rotation_dir_sim = path.join(home, 'Analysis', 'Efficiency', 'rotation_sim.json')
minimal_filename_dir = path.join(home, 'Analysis', 'minimal_path_length', 'minimal_filename_dict.json')

with open(translation_dir, 'r') as json_file:
    translation_dict = json.load(json_file)
    json_file.close()

with open(rotation_dir, 'r') as json_file:
    rotation_dict = json.load(json_file)
    json_file.close()

with open(minimal_filename_dir, 'r') as json_file:
    minimal_filename_dict = json.load(json_file)
    json_file.close()


# # __________ANTS ______ROT  +  TRANS_____________
#
# direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\without_cg_pathlength.json'
# with open(direct, 'r') as f:
#     data = json.load(f)
#
# translation_dict = {}
# rotation_dict = {}
#
# for filename in data.keys():
#     translation_dict[filename], rotation_dict[filename] = data[filename]
#
# minimal_translation = {'L': 12.73, 'M': 6.25, 'XL': 24.75, 'S (> 1)': 3.71, 'Single (1)': 3.71}
# minimal_rotation = {'L': 17.72, 'M': 8.66, 'XL': 34.66, 'S (> 1)': 4.28, 'Single (1)': 4.28}
#
# df_ant_excluded['minimal filename'] = df_ant_excluded['filename'].map(minimal_filename_dict)
# # create new column 'translation' which is from translation_dict
# df_ant_excluded['translation'] = pd.merge(pd.DataFrame(translation_dict.items(), )
# df_ant_excluded['rotation'] = df_ant_excluded['filename'].map(rotation_dict)
# df_ant_excluded['minimal translation'] = df_ant_excluded['size'].map(minimal_translation)
# df_ant_excluded['minimal rotation'] = df_ant_excluded['size'].map(minimal_rotation)
# df_ant_excluded['norm translation'] = df_ant_excluded['translation'] / df_ant_excluded['minimal translation']
# df_ant_excluded['norm rotation'] = df_ant_excluded['rotation'] / df_ant_excluded['minimal rotation']
#
# ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']
#
# fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 5))
# for size in ant_sizes[1:]:
#     df = df_ant_excluded[df_ant_excluded['size'] == size]
#     x_values = np.arange(0, df['norm translation'].max(), step=solver_step['ant'])
#     y_values = []
#
#     for x in x_values:
#         suc = df[(df['norm translation'] < x) & (df['winner'])]
#         y_values.append(len(suc) / len(df))
#     axs[0].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
#     axs[0].set_xlabel('norm translation')
#     axs[0].set_ylabel('fraction solved')
#
#     x_values = np.arange(0, df['norm rotation'].max(), step=solver_step['ant'])
#     y_values = []
#
#     for x in x_values:
#         suc = df[(df['norm rotation'] < x) & (df['winner'])]
#         y_values.append(len(suc) / len(df))
#     axs[1].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
#     axs[1].set_xlabel('norm rotation')
#
# axs[1].set_ylim([-0.00, 1.00])
# axs[0].set_ylim([-0.00, 1.00])
# plt.legend(prop={'size': 20})
# plt.savefig('Fig2_performance_ant.png')
# plt.savefig('Fig2_performance_ant.eps')
# plt.savefig('Fig2_performance_ant.svg')
# DEBUG = 1
#


# __________SIMULATION ______ROT  +  TRANS_____________
date = '2023_06_27'

scaling_factor = {'L': 1, 'M': 2, 'S': 4, 'XS': 8}


def scaled_time(filename):
    print(filename)
    x = get(filename)
    return len(x.angle) / 100 * scaling_factor[x.size]


df_sim = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
df_sim['time [s]'] = df_sim['filename'].map(lambda x: scaled_time(x))

fig, ax = plt.subplots(1, 1, figsize=(18 / 2.54, 5))
for size in ['XS', 'S', 'M', 'L']:
    df = df_sim[df_sim['size'] == size]
    x_values = np.arange(0, df['time [s]'].max(), step=solver_step['ant'])
    y_values = []

    for x in x_values:
        suc = df[df['time [s]'] < x]
        y_values.append(len(suc) / len(df))
    ax.step(x_values, y_values, label=size, color=colors[size], linewidth=2)
    ax.set_xlabel('scaled time [s]')
    ax.set_ylabel('fraction solved')

ax.set_ylim([-0.00, 1.00])
plt.legend(prop={'size': 20})
plt.savefig('Fig2_performance_sim.png')
plt.savefig('Fig2_performance_sim.eps')
plt.savefig('Fig2_performance_sim.svg')
DEBUG = 1

df_sim['minimal filename'] = df_sim['filename'].map(minimal_filename_dict)
df_sim['translation'] = df_sim['filename'].map(translation_dict)
df_sim['rotation'] = df_sim['filename'].map(rotation_dict)
df_sim['minimal translation'] = df_sim['minimal filename'].map(translation_dict_sim)
df_sim['minimal rotation'] = df_sim['minimal filename'].map(rotation_dict)
df_sim['norm translation'] = df_sim['translation'] / df_sim['minimal translation']
df_sim['norm rotation'] = df_sim['rotation'] / df_sim['minimal rotation']

ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']

fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 5))
for size in ant_sizes:
    df = df_ant_excluded[df_ant_excluded['size'] == size]
    x_values = np.arange(0, df['norm translation'].max(), step=solver_step['ant'])
    y_values = []

    for x in x_values:
        suc = df[(df['norm translation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    axs[0].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
    axs[0].set_xlabel('norm translation')
    axs[0].set_ylabel('fraction solved')

    x_values = np.arange(0, df['norm rotation'].max(), step=solver_step['ant'])
    y_values = []

    for x in x_values:
        suc = df[(df['norm rotation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    axs[1].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
    axs[1].set_xlabel('norm rotation')

axs[1].set_ylim([-0.00, 1.00])
axs[0].set_ylim([-0.00, 1.00])
plt.legend(prop={'size': 20})
plt.savefig('Fig2_performance_ant.png')
plt.savefig('Fig2_performance_ant.eps')
plt.savefig('Fig2_performance_ant.svg')
DEBUG = 1

# __________ HUMANS______ROT  +  TRANS_____________
for size, df in dfs_human.items():
    df['minimal filename'] = df['filename'].map(minimal_filename_dict)
    df['translation'] = df['filename'].map(translation_dict)
    df['rotation'] = df['filename'].map(rotation_dict)
    df['minimal translation'] = df['minimal filename'].map(translation_dict)
    df.loc[df['minimal translation'].isna(), 'minimal translation'] = df['minimal translation'].iloc[0]
    df['minimal rotation'] = df['minimal filename'].map(rotation_dict)
    df['norm translation'] = df['translation'] / df['minimal translation']
    df['norm rotation'] = df['rotation'] / df['minimal rotation']

fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 5))
for size, df in dfs_human.items():
    x_values = np.arange(0, df['norm translation'].max(), step=solver_step['human'])
    y_values = []

    for x in x_values:
        suc = df[(df['norm translation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    axs[0].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
    axs[0].set_xlabel('norm translation')
    axs[0].set_ylabel('fraction solved')

    x_values = np.arange(0, df['norm rotation'].max(), step=solver_step['human'])
    y_values = []

    for x in x_values:
        suc = df[(df['norm rotation'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    axs[1].step(x_values, y_values, label=size, color=colors[size], linewidth=2)
    axs[1].set_xlabel('norm rotation')

axs[1].set_ylim([-0.00, 1.00])
axs[0].set_ylim([-0.00, 1.00])
plt.legend(prop={'size': 20})
plt.savefig('Fig2_performance_human.png')
plt.savefig('Fig2_performance_human.eps')
plt.savefig('Fig2_performance_human.svg')
DEBUG = 1


# ___ MEANS ___
fig, ax = plt.subplots(figsize=(2.5, 2.5))
plt.xlabel('size')
plt.ylabel('norm translation')
df_size = df_ant_excluded.groupby('size')

means, stds = {}, {}
# plot the means of the normalized translation + rotation
for size in ant_sizes[1:]:
    df = df_ant_excluded[df_ant_excluded['size'] == size]
    df['path length'] = df['norm translation']
    df.loc[~df['winner'], 'path length'] = df.loc[~df['winner'], 'path length'] * 2
    # df = df[df['winner']]
    means[size] = np.mean(df['path length'])
    stds[size] = np.std(df['path length']) / np.sqrt(len(df))

plt.errorbar(means.keys(), means.values(), yerr=stds.values(), alpha=0.5)
plt.tight_layout()
plt.savefig('Fig2_inset_ants.png')
plt.savefig('Fig2_inset_ants.eps')
plt.savefig('Fig2_inset_ants.svg')
plt.close()
DEBUG = 1

# ___ MEANS ___
sizes = ['Small', 'Medium NC', 'Medium C', 'Large NC', 'Large C']

fig, ax = plt.subplots(figsize=(4, 2.5))
plt.xlabel('size')
plt.ylabel('norm translation')
df_size = df_ant_excluded.groupby('size')

means, stds = {}, {}

# plot the means of the normalized translation + rotation
for size in sizes:
    df = dfs_human[size]
    df['path length'] = df['norm translation']
    df.loc[~df['winner'], 'path length'] = df.loc[~df['winner'], 'path length'] * 2
    # df = df[df['winner']]
    means[size] = np.mean(df['path length'])
    stds[size] = np.std(df['path length']) / np.sqrt(len(df))

# x-axis labels on an angle


plt.errorbar(means.keys(), means.values(), yerr=stds.values(), alpha=0.5)
plt.tight_layout()
plt.savefig('Fig2_inset_human.png')
plt.savefig('Fig2_inset_human.eps')
plt.savefig('Fig2_inset_human.svg')
plt.close()
DEBUG = 1
