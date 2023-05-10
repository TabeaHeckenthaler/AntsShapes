import pandas as pd
from colors import colors_state
from PS_Search_Algorithms.human_network_simulation.StateMachine import HumanStateMachine
from Directories import network_dir
import os
import numpy as np
import json
from DataFrame.import_excel_dfs import dfs_human
from matplotlib import pyplot as plt
from Analysis.PathPy.Path import Path
from DataFrame.plot_dataframe import save_fig

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20


def plot_CDF(paths, ax, label=None, color=None, winner=None):
    measure = np.array([len(path) for path in paths])
    x_values = np.arange(0, np.max(measure)+2, step=1)
    y_values = []

    if winner is None:
        winner = np.ones(len(measure), dtype=bool)

    for x in x_values:
        suc = ((measure < x) & winner).sum()
        y_values.append(suc / len(measure))
    ax.step(x_values, y_values, label=label, color=color, linewidth=2)


def plot_bar_chart(paths, ax):
    for i, path in enumerate(paths):
        p = Path(time_step=60, time_series=path)
        p.bar_chart(ax=ax, axis_label=str(i), array=[state.strip('.') for state in path[:-1]], block=True)
    plt.subplots_adjust(hspace=.0)


def plot_state_percentages(df, ax, **kwargs):
    # find the percentage of every state in the state series
    states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'b1/b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
    state_percentages = pd.DataFrame(0, index=df['filename'], columns=states)
    for filename, ts in zip(df['filename'], df['state series']):
        state_percentages.loc[filename] = pd.Series(ts).value_counts(normalize=True)

    state_percentages['b1/b2'] = (state_percentages['b1'] + state_percentages['b2'])/2
    # change position column 'b1/b2' to come after 'ac'
    state_percentages.drop(columns=['b1', 'b2'], inplace=True)

    state_percentages.fillna(0, inplace=True)

    # find the average percentage of every state
    state_percentages_mean = state_percentages.mean()

    # plot state_percentages_mean empty bars
    ax.bar(state_percentages_mean.index, state_percentages_mean.values,
           color=[colors_state[state] for state in state_percentages_mean.index],
           edgecolor=[colors_state[state] for state in state_percentages_mean.index], **kwargs)


def plot_state_times(df, ax, **kwargs):
    # find the number of times every state appears in the state series
    states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'b1/b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
    state_times = pd.DataFrame(0, index=df['filename'], columns=states)
    for filename, ts in zip(df['filename'], df['state series']):
        state_times.loc[filename] = pd.Series(ts).value_counts()

    state_times.fillna(0, inplace=True)
    # state_times['b1/b2'] = (state_times['b1'] + state_times['b2'])/2
    state_times['b1/b2'] = np.logical_or(state_times['b1'], state_times['b2'])
    # change position column 'b1/b2' to come after 'ac'
    state_times.drop(columns=['b1', 'b2'], inplace=True)

    # find the average percentage of every state
    state_times_mean = state_times.mean()
    # error
    state_times_error = state_times.std() / np.sqrt(df.shape[0])

    # plot state_percentages_mean empty bars
    ax.bar(state_times_mean.index, state_times_mean.values,
           color=[colors_state[state] for state in state_times_mean.index],
           edgecolor=[colors_state[state] for state in state_times_mean.index],
           yerr=state_times_error.values,
           **kwargs)
    ax.set_ylabel('<# times state was accessed per exp.>')
    ax.set_xlabel('states')


def plot_state_accessed(df, ax, **kwargs):
    # find the number of times every state appears in the state series
    states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'b1/b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
    state_times = pd.DataFrame(0, index=df['filename'], columns=states)
    for filename, ts in zip(df['filename'], df['state series']):
        state_times.loc[filename] = pd.Series(ts).value_counts().astype(bool).astype(int)

    state_times.fillna(0, inplace=True)
    # state_times['b1/b2'] = (state_times['b1'] + state_times['b2']) / 2
    state_times['b1/b2'] = np.logical_or(state_times['b1'], state_times['b2'])
    # change position column 'b1/b2' to come after 'ac'
    state_times.drop(columns=['b1', 'b2'], inplace=True)

    # find the average percentage of every state
    state_times_mean = state_times.mean()
    # error
    state_times_error = state_times.std() / np.sqrt(df.shape[0])

    # plot state_percentages_mean empty bars
    ax.bar(state_times_mean.index, state_times_mean.values,
           color=[colors_state[state] for state in state_times_mean.index],
           edgecolor=[colors_state[state] for state in state_times_mean.index],
           yerr=state_times_error.values,
           **kwargs)
    ax.set_ylabel('<exp. that accessed state>')
    ax.set_xlabel('states')


# ================== import simulation human ==================
from PS_Search_Algorithms.human_network_simulation.StateMachine import name
# name = 'pattern_recognition_0.3_ineg_bias_2_firstStep'

df_sim = HumanStateMachine.df(name=name + "states_small.json")
df_sim.rename(columns={'name': 'filename', 'states_series': 'state series'}, inplace=True)

# ================== import experiment human Small ==================
with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
    state_series_dict = json.load(json_file)
    json_file.close()

size = 'Large NC'  # ['Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']
df_exp = dfs_human[size]
df_exp['state series'] = df_exp['filename'].map(state_series_dict)

# # ================== PERCENTAGES ==================
fig_percentages, ax_percentages = plt.subplots()
plot_state_percentages(df_sim, ax_percentages, fill=False, linewidth=4, label='simulation')
plot_state_percentages(df_small_exp, ax_percentages, alpha=0.3, label='experiment')
plt.legend()
save_fig(fig_percentages, name='percentages_' + name)

# ================== TIMES ==================
fig_times, ax_times = plt.subplots()
plot_state_times(df_sim, ax_times, fill=False, linewidth=4, label='simulation')
plot_state_times(df_exp, ax_times, alpha=0.3, label='experiment ' + size)
plt.legend()
save_fig(fig_times, name=size + '_times_' + name)

# ================== ACCESSED STATES ==================
fig_times, ax_times = plt.subplots()
plot_state_accessed(df_sim, ax_times, fill=False, linewidth=4, label='simulation')
plot_state_accessed(df_exp, ax_times, alpha=0.3, label='experiment ' + size)
plt.legend()
save_fig(fig_times, name=size + '_accessed_' + name)

# ================== BAR PLOTS ==================
fig_bar, axs_bar = plt.subplots(2)
fig_bar.set_size_inches(20, 10)
plot_bar_chart(df_sim['state series'], axs_bar[0])
axs_bar[0].set_title('simulation')

plot_bar_chart(df_exp['state series'], axs_bar[1])
axs_bar[1].set_title('experiment ' + size)
fig_bar.tight_layout()
save_fig(fig_bar, name=size + '_bar_chart_' + name)

# ================== CDF PLOTS ==================
fig_bar, ax = plt.subplots()
fig_bar.set_size_inches(5, 5)
plot_CDF(df_sim['state series'], ax, label='simulation')
plot_CDF(df_exp['state series'], ax, label='experiment ' + size)
plt.legend()
save_fig(fig_bar, name=size + '_CDF_' + name)
