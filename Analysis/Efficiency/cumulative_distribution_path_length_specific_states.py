import numpy as np
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import df_ant, dfs_ant, df_pheidole, dfs_pheidole, df_human, dfs_human
from DataFrame.plot_dataframe import save_fig
from os import path
from Directories import home
import json
from colors import colors_humans as colors

direct_pre = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\pre_c_pathlength.json'
direct_post = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\post_c_pathlength.json'
direct_post_without = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\post_c_without_c_pathlength.json'
translation_dir = path.join(home, 'Analysis', 'Efficiency', 'translation.json')
rotation_dir = path.join(home, 'Analysis', 'Efficiency', 'rotation.json')
minimal_filename_dict = path.join(home, 'Analysis', 'minimal_path_length', 'minimal_filename_dict.json')

with open(translation_dir, 'r') as json_file:
    translation_dict = json.load(json_file)
    json_file.close()

with open(rotation_dir, 'r') as json_file:
    rotation_dict = json.load(json_file)
    json_file.close()

with open(direct_pre, 'r') as json_file:
    pre_c_dict = json.load(json_file)
    json_file.close()

with open(direct_post, 'r') as json_file:
    post_c_dict = json.load(json_file)
    json_file.close()

with open(direct_post_without, 'r') as json_file:
    post_c_without_c_dict = json.load(json_file)
    json_file.close()

with open(minimal_filename_dict, 'r') as json_file:
    minimal_filename_dict = json.load(json_file)
    json_file.close()

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}

solver = 'ant'

df_solver, dfs_solver = {'human': (df_human, dfs_human),
                         'ant': (df_ant, dfs_ant),
                         'pheidole': (df_pheidole, dfs_pheidole)}[solver]

df_solver['minimal filename'] = df_solver['filename'].map(minimal_filename_dict)
df_solver['pre_c'] = df_solver['filename'].map(pre_c_dict)
df_solver['post_c'] = df_solver['filename'].map(post_c_dict)
df_solver['post_c_without_c'] = df_solver['filename'].map(post_c_without_c_dict)
df_solver['minimal filename'] = df_solver['filename'].map(minimal_filename_dict)
df_solver['minimal translation'] = df_solver['minimal filename'].map(translation_dict)
df_solver['minimal rotation'] = df_solver['minimal filename'].map(rotation_dict)  # this is already normalized
df_solver['minimal path length'] = df_solver['minimal translation'] + df_solver['minimal rotation']
df_solver['norm pre_c'] = df_solver['pre_c'] / df_solver['minimal path length']
df_solver['norm post_c'] = df_solver['post_c'] / df_solver['minimal path length']
df_solver['norm post_c_without_c'] = df_solver['post_c_without_c'] / df_solver['minimal path length']

fig, ax = plt.subplots()
plt.xlabel('normalized pL pre_c')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]
    x_values = np.arange(0, df['norm pre_c'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm pre_c'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
save_fig(fig, solver + 'cum_distribution_path_length_pre_c')


fig, ax = plt.subplots()
plt.xlabel('normalized pL post_c')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]
    x_values = np.arange(0, df['norm post_c'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm post_c'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.legend(prop={'size': 20})
save_fig(fig, solver + 'cum_distribution_path_length_post_c')


fig, ax = plt.subplots()
plt.xlabel('normalized pL post_c_without_c')
plt.ylabel('percentage solved')
plt.show(block=0)
plt.title(solver + ' CDF')

for size, df_size in dfs_solver.items():
    df = df_solver[df_solver['filename'].isin(df_size['filename'])]
    x_values = np.arange(0, df['norm post_c'].max(), step=solver_step[solver])
    y_values = []

    for x in x_values:
        suc = df[(df['norm post_c_without_c'] < x) & (df['winner'])]
        y_values.append(len(suc) / len(df))
    plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2)

plt.ylim([-0.05, 1.05])
plt.xlim([-0.5, 10])
plt.legend(prop={'size': 20})

save_fig(fig, solver + 'cum_distribution_path_length_post_c_without_c')

DEBUG = 1