from Analysis.PathPy.Path import time_series_dict, Path
from trajectory_inheritance.exp_types import solver_geometry, exp_types
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import pandas as pd
from Analysis.PathPy.SPT_states import all_states
import numpy as np
from DataFrame.plot_dataframe import save_fig
from DataFrame.Altered_DataFrame import Altered_DataFrame
from Analysis.average_carrier_number.averageCarrierNumber import myDataFrame
from DataFrame.food_in_the_back import myDataFrame as df_food
from Analysis.Path_Length_States import exp_day
from Analysis.GeneralFunctions import flatten


def find_distribution(time_series):
    b = Path.time_stamped_series(time_series, 0.25)
    array = np.array(b)
    state_dict = {}
    for state in np.unique(array[:, 0]):
        state_dict[state] = list(array[array[:, 0] == state][:, 1].astype(float))
    return state_dict


def plot_state_distribution(axs, dist):
    if axs.ndim > 1:
        axs = flatten(axs)
    for (state, distr_of_state), ax in zip(dist.items(), axs):
        range = None
        # if state in ['a', 'b', 'c', 'e']:
        #     range = [0, 300]
        ax.hist(distr_of_state, range=range, bins=20)
        ax.set_title(state)


if __name__ == '__main__':
    myDataFrame = Altered_DataFrame(myDataFrame)
    DEBUG = 1
    myDataFrame.df['food in back'] = df_food.df['food in back']
    myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict)

    solver = 'ant'
    shape = 'SPT'

    plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}
    dfss = myDataFrame.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver], shape=shape,
                                                geometry=solver_geometry[solver], initial_cond='back')

    fig, axes = plt.subplots(len(dfss) * 2, 6)
    fig.set_size_inches(20, 3 * len(dfss) * 2, forward=True)
    i = 0

    for size, dfs in dfss.items():
        for sep, df in dfs.items():
            d_all_exp = {s: [] for s in all_states}

            for filename in df['filename']:
                p = Path.symmetrize(Path.only_states(time_series_dict[filename]))
                d_exp = find_distribution(p)
                for state in d_exp.keys():
                    d_all_exp[state] = d_all_exp[state] + d_exp[state]

            for state in [key for key, value in d_all_exp.items() if len(value) < 1] + ['i']:
                if state in d_all_exp.keys():
                    d_all_exp.pop(state)

            sep_title_dict = {'non_communication': 'NC', 'communication': 'C'}
            if sep in sep_title_dict.keys():
                sep_title = sep_title_dict[sep]
            else:
                sep_title = sep
            axes[i][0].set_ylabel(size + ' ' + sep_title)
            plot_state_distribution(axes[i], d_all_exp)

            i += 1

    plt.tight_layout()
    save_fig(fig, solver + 'state_distributions')
