from Analysis.PathPy.Path import Path
from Analysis.PathPy.SPT_states import states, allowed_transition_attempts, forbidden_transition_attempts
import os
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
import pathpy as pp
from Directories import network_dir
from DataFrame.plot_dataframe import save_fig
import json
from trajectory_inheritance.exp_types import exp_types
import csv
from DataFrame.Altered_DataFrame import choose_trajectories, Altered_DataFrame
import numpy as np
from trajectory_inheritance.trajectory import get, solver_geometry
import pandas as pd
from matplotlib import pyplot as plt


def flatten(t):
    return [item for sublist in t for item in sublist]


class Paths(pp.Paths):
    def __init__(self, solver, size, shape, geometry, time_step=0.25, communication=None):
        super().__init__()
        if size is not None and 'Small' in size:
            size = 'Small'
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry
        self.time_step = time_step
        self.time_series = None
        self.single_paths = None
        self.max_subpath_length = 2
        self.communication = communication

    def save_dir(self, name=None):
        if name is None:
            name = '_'.join(['paths', self.solver, self.size, self.shape])
        if self.communication is not None:
            name = name + '_comm_' + str(self.communication)
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_transitions')

    def load_paths(self, filenames=None, only_states=False):
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
        if os.path.exists(self.save_dir() + '.txt'):
            with open(self.save_dir() + '.txt', 'r') as json_file:
                self.time_series = json.load(json_file)
                if filenames is not None:
                    self.time_series = dictfilt(self.time_series, filenames)
                if only_states:
                    self.time_series = {name: [state[0] for state in series] for name, series in
                                        self.time_series.items()}
                self.single_paths = {name: Path(self.time_step, time_s) for name, time_s in self.time_series.items()}

        elif self.time_series is None:
            trajectories = choose_trajectories(solver=self.solver, size=self.size, shape=self.shape,
                                               geometry=self.geometry, communication=self.communication)
            self.time_series = self.calculate_time_series(trajectories, only_states=only_states)
            self.save_paths()
            if filenames is not None:
                self.time_series = dictfilt(self.time_series, filenames)
            if only_states:
                self.time_series = {name: [state[0] for state in series] for name, series in
                                    self.time_series.items()}
        self.add_paths()

    def add_paths(self):
        [self.add_path(p) for p in self.time_series.values()]

    def calculate_time_series(self, trajectories, only_states=False) -> dict:
        self.single_paths = {}
        cs_labeled = None
        for i, x in enumerate(trajectories):
            if x.size == 'Small':
                size = 'Small Far'
            else:
                size = x.size
            print(i)
            if cs_labeled is None or cs_labeled.size != x.size:
                cs_labeled = ConfigSpace_Labeled(x.solver, size, x.shape, self.geometry)
                cs_labeled.load_labeled_space()
            self.single_paths[x.filename] = Path(self.time_step, conf_space_labeled=cs_labeled, x=x,
                                                 only_states=only_states)
        return {name: path.time_series for name, path in self.single_paths.items()}

    def save_paths(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir()
        json_string = self.time_series
        with open(save_dir + '.txt', 'w') as json_file:
            json.dump(json_string, json_file)
        print('Saved paths in ', save_dir)


class PathsTimeStamped(Paths):
    def __init__(self, solver, size, shape, geometry, time_step=0.25, communication=None):
        super().__init__(solver, size, shape, geometry, communication=communication)
        self.time_series = None
        self.time_stamped_series = None
        self.time_step = time_step

    @staticmethod
    def measure_in_state(time_stamped_series: list):
        so = flatten(time_stamped_series)
        average_time = {state: [] for state in states + allowed_transition_attempts + forbidden_transition_attempts}
        average_time.pop('0')
        for state, time in so:
            average_time[state].append(time)
        return average_time

    def calculate_timestamped(self):
        if self.time_series is None:
            self.load_paths()
        return {name: Path.time_stamped_series(time_series, self.time_step)
                for name, time_series in self.time_series.items()}

    def calculate_path_length_stamped(self):
        if self.time_series is None:
            self.load_paths()
        DEBUG = 1
        return {name: Path.path_length_stamped_series(name, time_series, self.time_step)
                for name, time_series in self.time_series.items()}

    def save_dir_timestamped(self, name=None):
        if name is None:
            name = '_'.join([self.solver, self.size, self.shape])
        if self.communication is not None:
            name = name + '_comm_' + str(self.communication)
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_states_timestamped')

    def save_timestamped_paths(self, save_dir_timestamped=None):
        json_string = self.time_stamped_series
        if save_dir_timestamped is None:
            save_dir_timestamped = self.save_dir_timestamped()
        with open(save_dir_timestamped + '.txt', 'w') as json_file:
            json.dump(json_string, json_file)
        print('Saved timestamped paths in ', save_dir_timestamped)

    def load_time_stamped_paths(self, filenames=None):
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
        if os.path.exists(self.save_dir_timestamped() + '.txt'):
            with open(self.save_dir_timestamped() + '.txt', 'r') as json_file:
                self.time_stamped_series = json.load(json_file)
                if filenames is not None:
                    self.time_stamped_series = dictfilt(self.time_stamped_series, filenames)
        else:
            print('Calculating paths')
            self.time_stamped_series = self.calculate_timestamped()
            self.save_timestamped_paths()
            if filenames is not None:
                self.time_stamped_series = dictfilt(self.time_stamped_series, filenames)

    def save_csv(self, save_dir_timestamped=None):
        if save_dir_timestamped is None:
            save_dir_timestamped = self.save_dir_timestamped()

        print('Saved timestamped paths in ', save_dir_timestamped + '.csv')
        with open(save_dir_timestamped + '.csv', 'w') as f:
            rows = [[i] + flatten(series) for i, series in enumerate(self.time_stamped_series.values())]
            write = csv.writer(f, delimiter=" ")
            write.writerows(rows)


class PathWithoutSelfLoops(Paths):
    def add_paths(self):
        [self.add_path(p.state_series) for p in self.single_paths.values()]


def humans():
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    # communication = [True, False]

    for size in exp_types[shape][solver]:
        # for com in communication:
        paths = PathsTimeStamped(solver, size, shape, geometry, communication=None)
        paths.load_paths()
        paths.load_time_stamped_paths()
        paths.save_csv()
        DEBUG = 1
    # filename = list(paths.time_series.keys())[0]
    # x = get(filename)
    # x.play(path=paths.single_paths[filename], videowriter=True, step=2)
    # DEBUG = 1


def perfect_human():
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    # communication = [True, False]

    filenames = ['large_20210805171741_20210805172610_perfect',
                 'medium_20210507225832_20210507230303_perfect',
                 'small2_20220308120548_20220308120613_perfect']

    trajectories = [get(filename) for filename in filenames]

    paths = PathsTimeStamped(solver, None, shape, geometry, communication=None)
    paths.time_series = paths.calculate_time_series(trajectories)
    paths.time_stamped_series = paths.calculate_timestamped()

    name = 'perfect_human_paths'
    paths.save_paths(paths.save_dir(name=name))
    paths.save_timestamped_paths(paths.save_dir_timestamped(name=name))
    paths.save_csv(paths.save_dir_timestamped(name=name))


def ants():
    solver, shape, geometry = 'ant', 'SPT', (
    'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')

    # for size in exp_types[shape][solver]:
    for size in ['XL']:
        paths = PathsTimeStamped(solver, size, shape, geometry)

        paths.load_paths()
        paths.load_time_stamped_paths()
        # paths.save_csv()

    # filename = list(paths.time_series.keys())[0]
    # x = get(filename)
    # x.play(path=paths.single_paths[filename], videowriter=True, step=2)
    # DEBUG = 1


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
        dfs = df.get_separate_data_frames(solver, solvers[solver], shape='SPT', geometry=solver_geometry[solver],
                                          initial_cond='back')

        for key, d in dfs.items():
            experiments_df = pd.concat(d)
            filenames = experiments_df['filename']
            if len(filenames) > 0:
                size = get_size(experiments_df)
                paths = PathsTimeStamped(solver, size, shape, solver_geometry[solver])
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
                                                   in average_time_in_states[solver][key].items() if len(time_list) > 0}
        fig = plot_dict(average_time_in_states[solver])
        plt.ylabel('average time spent in state')
        save_fig(fig, solver + 'average_time_spent_in_states')


def av_path_in_states():
    solvers = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}}
    shape = 'SPT'
    average_path_in_states = {}

    for solver in solvers.keys():
        average_path_in_states[solver] = {}

        df = Altered_DataFrame()
        dfs = df.get_separate_data_frames(solver, solvers[solver], shape='SPT', geometry=solver_geometry[solver],
                                          initial_cond='back')

        for key, d in dfs.items():
            experiments_df = pd.concat(d)
            filenames = experiments_df['filename']
            if len(filenames) > 0:
                size = get_size(experiments_df)
                paths = PathsTimeStamped(solver, size, shape, solver_geometry[solver])
                paths.load_paths(only_states=True)
                paths.time_stamped_series = paths.calculate_path_length_stamped()

                path_length_stamped_states = []
                for filename in filenames:
                    if filename in paths.time_stamped_series.keys():
                        path_length_stamped_states.append(paths.time_stamped_series[filename])
                    else:
                        print(filename)
                average_path_in_states[solver][key] = PathsTimeStamped.measure_in_state(path_length_stamped_states)

            average_path_in_states[solver][key] = {state: np.mean(pathlength_list)
                                                          /experiments_df['minimal path length [length unit]'].iloc[0]
                                                   for state, pathlength_list
                                                   in average_path_in_states[solver][key].items() if len(pathlength_list) > 0}
        fig = plot_dict(average_path_in_states[solver])
        plt.ylabel('average path length walked in state/minimal total path length')
        save_fig(fig, solver + 'average_path_spent_in_states')


def plot_dict(dic: dict):
    df = pd.DataFrame(dic)
    df = df.transpose()
    df.plot(kind="bar", stacked=True)
    return plt.gcf()


if __name__ == '__main__':
    av_time_in_states()
    av_path_in_states()
    # ants()
    # humans()
    # perfect_human()
