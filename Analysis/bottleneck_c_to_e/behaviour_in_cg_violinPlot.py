from DataFrame.import_excel_dfs import df_minimal, dfs_ant, dfs_ant_old
from DataFrame.gillespie_dataFrame import dfs_gillespie
import numpy as np
import json
from DataFrame.plot_dataframe import save_fig
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})

solver, shape = 'gillespie', 'SPT'
df_dict_sep_by_size = dfs_gillespie
# x = get(df_dict_sep_by_size['S'].iloc[0]['filename'])
# x.play()
radius = 3
_add = '_gillespie' + str(radius)

# solver, shape = 'ant', 'SPT'
# radius = 10
# _add = '_ant' + str(radius)
# df_dict_sep_by_size = {size: pd.concat([dfs_ant[size], dfs_ant_old[size]]) for size in dfs_ant.keys()}

with open('results\\pL_in_c' + _add + '.json', 'r') as f:
    pL_in_c = json.load(f)
with open('results\\pL_out_c' + _add + '.json', 'r') as f:
    pL_out_c = json.load(f)
with open('results\\pL_after_first_e' + _add + '.json', 'r') as f:
    pL_after_first_e = json.load(f)
with open('results\\times_entered_to_c' + _add + '.json', 'r') as f:
    times_entered_to_c = json.load(f)
with open('results\\distance_in_c_on_edge' + _add + '.json', 'r') as f:
    distance_in_c_on_edge = json.load(f)
with open('results\\distance_in_c_off_edge' + _add + '.json', 'r') as f:
    distance_in_c_off_edge = json.load(f)

pL_in_c_size = {}
pL_out_c_size = {}
pL_after_first_e_size = {}
times_entered_to_c_size = {}
distance_in_c_on_edge_size = {}
distance_in_c_off_edge_size = {}
on_edge_fraction_size = {}


for size, df in df_dict_sep_by_size.items():
    print(size)
    minimal = df_minimal[(df_minimal['size'] == df.iloc[0]['size']) & (df_minimal['shape'] == 'SPT') & (
            df_minimal['initial condition'] == 'back')].iloc[0]['path length [length unit]']

    df['pL_in_c'] = df['filename'].map(pL_in_c)
    df['pL_out_c'] = df['filename'].map(pL_out_c)
    df['pL_after_first_e'] = df['filename'].map(pL_after_first_e)
    df['times_entered_to_c'] = df['filename'].map(times_entered_to_c)
    df['distance_in_c_on_edge'] = df['filename'].map(distance_in_c_on_edge).apply(np.sum)
    df['distance_in_c_off_edge'] = df['filename'].map(distance_in_c_off_edge).apply(np.sum)

    pL_in_c_size[size] = [item / minimal for sublist in df['pL_in_c'].dropna() for item in sublist]
    pL_out_c_size[size] = [item / minimal for sublist in df['pL_out_c'].dropna() for item in sublist]
    pL_after_first_e_size[size] = [item / minimal for item in df['pL_after_first_e'].dropna()]
    times_entered_to_c_size[size] = df['times_entered_to_c'].dropna().tolist()
    distance_in_c_on_edge_size[size] = df['distance_in_c_on_edge'].dropna() / minimal.tolist()
    distance_in_c_off_edge_size[size] = df['distance_in_c_off_edge'].dropna() / minimal.tolist()
    df['on_edge_fraction'] = df['distance_in_c_on_edge'] / (
            df['distance_in_c_on_edge'] + df['distance_in_c_off_edge'])
    on_edge_fraction_size[size] = df['on_edge_fraction'].dropna().tolist()

dataset = [v for v in pL_in_c_size.values()]
positions = [k for k in range(len(dataset))]
fig_pL_in_c, axes = plt.subplots()
plt.ylabel('scaled path length in c [pL/minimal solving pL]')
plt.title('pL in c per entrance to c')

plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(pL_in_c_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

dataset = [v for v in pL_out_c_size.values()]
positions = [k for k in range(len(dataset))]
fig_pL_out_c, axes = plt.subplots()
plt.ylabel('scaled path length outside of c [pL/minimal solving pL]')
plt.title('pL outside of c in between entrances to c')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(pL_out_c_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

dataset = [v for v in pL_after_first_e_size.values()]
positions = [k for k in range(len(dataset))]
fig_pL_after_first_e, axes = plt.subplots()
plt.ylabel('scaled path length after entering e the first time [pL/minimal solving pL]')
plt.title('pL path length after entering e the first time')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(pL_out_c_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)
axes.set_ylim([0, 10])

dataset = [v for v in times_entered_to_c_size.values()]
positions = [k for k in range(len(dataset))]
fig_times_entered_to_c, axes = plt.subplots()
plt.ylabel('times entered to c in successful and unsuccessful experiments')
plt.title('times entered to c')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(times_entered_to_c_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

dataset = [v for v in distance_in_c_on_edge_size.values()]
positions = [k for k in range(len(dataset))]
fig_distance_in_c_on_edge, axes = plt.subplots()
plt.ylabel('scaled path length')
plt.title('distance_in_c_on_edge per experiment')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(distance_in_c_on_edge_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

dataset = [v for v in distance_in_c_off_edge_size.values()]
positions = [k for k in range(len(dataset))]
fig_distance_in_c_off_edge, axes = plt.subplots()
plt.ylabel('scaled path length')
plt.title('distance_in_c_off_edge per experiment')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(distance_in_c_off_edge_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

dataset = [v for v in on_edge_fraction_size.values()]
positions = [k for k in range(len(dataset))]
fig_on_edge_fraction, axes = plt.subplots()
plt.ylabel('on_edge_fraction')
plt.title('on_edge_fraction')
plt.xlabel('size')
axes.set_xticks(positions)
axes.set_xticklabels(on_edge_fraction_size.keys())
parts = axes.violinplot(dataset=dataset, positions=positions, showmedians=1, showextrema=0)

save_fig(fig_pL_in_c, 'pL_in_c_violin' + _add)
save_fig(fig_pL_out_c, 'pL_out_c_violin' + _add)
save_fig(fig_pL_after_first_e, 'pL_after_first_e_violin' + _add)
save_fig(fig_times_entered_to_c, 'times_entered_to_c_violin' + _add)
save_fig(fig_distance_in_c_on_edge, 'distance_in_c_on_edge_violin' + _add)
save_fig(fig_distance_in_c_off_edge, 'distance_in_c_off_edge_violin' + _add)
save_fig(fig_on_edge_fraction, 'on_edge_fraction_violin' + _add)
DEBUG = 1
