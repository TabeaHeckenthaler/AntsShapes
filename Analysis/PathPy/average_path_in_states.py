from Analysis.PathPy.Paths import plot_separately, PathsTimeStamped
from DataFrame.plot_dataframe import save_fig
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.exp_types import solver_geometry
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_dict(dic: dict, ax):
    df = pd.DataFrame(dic)
    df = df.transpose()
    df.plot(kind="bar", stacked=True, ax=ax)
    return plt.gcf()


def get_size(df):
    sizes = set(df['size'])

    if 'Small Far' in sizes:
        sizes.remove('Small Far')
        sizes.add('Small')

    if 'Small Near' in sizes:
        sizes.remove('Small Near')
        sizes.add('Small')

    if len(sizes) == 1:
        return sizes.pop()
    else:
        raise ValueError('more than one size')


def av_time_in_states():
    solvers = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}}
    shape = 'SPT'

    average_time_in_states = {}

    for solver in solvers.keys():
        average_time_in_states[solver] = {}

        df = Altered_DataFrame()
        dfs = df.get_separate_data_frames(solver, plot_separately[solver], shape='SPT', geometry=solver_geometry[solver],
                                          initial_cond='back')

        for key, ds in dfs.items():
            # experiments_df = pd.concat(ds)
            for experiments_df in ds:
                filenames = experiments_df['filename']
                if len(filenames) > 0:
                    size = get_size(experiments_df)
                    paths = PathsTimeStamped(solver, shape, solver_geometry[solver], size)
                    paths.load_paths(only_states=True)
                    paths.time_stamped_series = paths.calculate_timestamped()

                    time_stamped_states = []
                    for filename in filenames:
                        if filename in paths.time_stamped_series.keys():
                            time_stamped_states.append(paths.time_stamped_series[filename])
                        else:
                            print(filename)
                    average_time_in_states[solver][key] = PathsTimeStamped.measure_in_state(time_stamped_states)

                average_time_in_states[solver][key] = {state: np.mean(time_list) for state, time_list
                                                       in average_time_in_states[solver][key].items() if
                                                       len(time_list) > 0}
            fig = plot_dict(average_time_in_states[solver])
            plt.ylabel('average time spent in state')
            save_fig(fig, solver + 'average_time_spent_in_states')


def av_path_in_states():
    solvers = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}}
    shape = 'SPT'
    average_path_in_states = {}

    # for solver in solvers.keys():
    for solver in ['human']:
        average_path_in_states[solver] = {}

        df = Altered_DataFrame()
        dfs = df.get_separate_data_frames(solver, plot_separately[solver], shape='SPT', geometry=solver_geometry[solver],
                                          initial_cond='back')

        for size_split, list_of_dataframes in dfs.items():
            # experiments_df = pd.concat(list_of_dataframes)
            for splitter, experiments_df in list_of_dataframes.items():
                if splitter not in average_path_in_states[solver].keys():
                    average_path_in_states[solver][splitter] = {}
                filenames = experiments_df['filename']
                # if "large_20211006174255_20211006174947" in filenames:
                #     DEBUG = 1
                if len(filenames) > 0:
                    size = get_size(experiments_df)
                    paths = PathsTimeStamped(solver, shape, solver_geometry[solver], size)
                    paths.load_paths(only_states=True, filenames=filenames)
                    paths.time_stamped_series = paths.calculate_path_length_stamped()

                    path_length_stamped_states = []
                    for filename in filenames:
                        if filename in paths.time_stamped_series.keys():
                            path_length_stamped_states.append(paths.time_stamped_series[filename])
                        else:
                            print(filename)
                    paths_in_states = {state: pathlength_list for state, pathlength_list in
                                       PathsTimeStamped.measure_in_state(path_length_stamped_states).items()
                                       if len(pathlength_list) > 0}
                    minimal_path = experiments_df['minimal path length [length unit]'].iloc[0]
                    average_path_in_states[solver][splitter][size_split] = {state: np.mean(pathlength_list) / minimal_path
                                                                            for state, pathlength_list in paths_in_states.items()}

        fig, axs = plt.subplots(ncols=len(average_path_in_states[solver].keys()))
        for ax, splitter in zip(axs, average_path_in_states[solver].keys()):
            plot_dict(average_path_in_states[solver][splitter], ax)
            plt.ylabel('average path length walked in state/minimal total path length')
            ax.title.set_text(splitter)
        axs[-1].get_legend().remove()
        save_fig(fig, solver + 'average_path_spent_in_states')