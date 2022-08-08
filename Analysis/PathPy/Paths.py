from Analysis.PathPy.Path import Path
from Analysis.PathPy.SPT_states import states, allowed_transition_attempts, forbidden_transition_attempts
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
import pathpy as pp
from Directories import network_dir
from DataFrame.Altered_DataFrame import choose_trajectories
from trajectory_inheritance.exp_types import exp_types
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_human import perfect_filenames
import os
import json
import csv


# TODO: This whole module is really really useless, if I have DataFrame. Skim through it and delete most of it...

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}


def flatten(t):
    return [item for sublist in t for item in sublist]


class Paths(pp.Paths):
    def __init__(self, solver, shape, geometry, time_step=0.25, size=None, communication=None):
        super().__init__()
        if size is not None and 'Small' in size:
            size = 'Small'
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry
        self.time_step = time_step
        self.time_series = {}
        self.single_paths = {}
        self.max_subpath_length = 2
        self.communication = None

    def save_dir(self, name=None, size=None):
        if size is None:
            size = self.size
        if name is None:
            name = '_'.join(['paths', self.solver, size, self.shape])
        # if self.communication is not None:
        #     name = name + '_comm_' + str(self.communication)
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_transitions')

    def load_paths(self, filenames=None, only_states=False, symmetric_states=False, save_dir=None, size=None):

        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

        if self.size is None and size is None:
            for size in exp_types[self.shape][self.solver]:
                self.load_paths(save_dir=self.save_dir(size=size), size=size, filenames=filenames,
                                only_states=only_states, symmetric_states=symmetric_states)

        else:
            if save_dir is None:
                save_dir = self.save_dir()

            if os.path.exists(save_dir + '.txt'):
                with open(save_dir + '.txt', 'r') as json_file:
                    time_series = json.load(json_file)
                    if filenames is not None:
                        time_series = dictfilt(time_series, filenames)
                    if only_states:
                        time_series = {name: [state[0] for state in series] for name, series in time_series.items()}
                    if symmetric_states:
                        time_series = {name: [state.replace('d', 'e') for state in series]
                                       for name, series in time_series.items()}
                    self.time_series.update(time_series)
                    self.single_paths = {name: Path(self.time_step, time_s) for name, time_s in self.time_series.items()}

            else:
                trajectories = choose_trajectories(solver=self.solver, size=self.size, shape=self.shape,
                                                   geometry=self.geometry, communication=self.communication)
                time_series = self.calculate_time_series(trajectories, only_states=only_states)
                self.save_paths()
                if filenames is not None:
                    time_series = dictfilt(time_series, filenames)
                if only_states:
                    time_series = {name: [state[0] for state in series] for name, series in time_series.items()}
                if symmetric_states:
                    time_series = {name: [state.replace('d', 'e') for state in series] for name, series in time_series.items()}
                self.time_series.update(time_series)

            self.add_paths(time_series)

    def add_paths(self, time_series):
        [self.add_path(p) for p in time_series.values()]

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
    def __init__(self, solver, shape, geometry, size=None, time_step=0.25):
        super().__init__(solver, shape, geometry, size=size)
        self.time_series = {}
        self.time_stamped_series = {}
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
        if len(self.time_series) == 0:
            self.load_paths()
        return {name: Path.time_stamped_series(time_series, self.time_step)
                for name, time_series in self.time_series.items()}

    def calculate_path_length_stamped(self):
        if len(self.time_series) == 0:
            self.load_paths()
        return {name: Path.path_length_stamped_series(name, time_series, self.time_step)
                for name, time_series in self.time_series.items()}

    def save_dir_timestamped(self, name=None):
        if name is None:
            name = '_'.join([self.solver, self.size, self.shape])
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
    def add_paths(self, _):
        # [self.add_path(p) for p in time_series.values()]
        [self.add_path(p.state_series) for p in self.single_paths.values()]


def perfect_human():
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    # communication = [True, False]

    trajectories = [get(filename) for filename in perfect_filenames]

    paths = PathsTimeStamped(solver, shape, geometry)
    paths.time_series = paths.calculate_time_series(trajectories)
    paths.time_stamped_series = paths.calculate_timestamped()

    name = 'perfect_human_paths'
    paths.save_paths(paths.save_dir(name=name))
    paths.save_timestamped_paths(paths.save_dir_timestamped(name=name))
    paths.save_csv(paths.save_dir_timestamped(name=name))


def run(solver, shape, geometry):
    paths = PathsTimeStamped(solver, shape, geometry)
    paths.load_paths()
    paths.load_time_stamped_paths()
    paths.save_paths()
    paths.save_timestamped_paths()
    paths.save_csv()


def all2():
    shape = 'SPT'
    solver, geometry = 'human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    run(solver, shape, geometry)

    solver, geometry = 'ant', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')
    run(solver, shape, geometry)

    solver, geometry = 'humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')
    run(solver, shape, geometry)


if __name__ == '__main__':
    D = 1