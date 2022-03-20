from Analysis.PathPy.Path import Path
import os
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
import pathpy as pp
from Directories import network_dir
import json
from trajectory_inheritance.exp_types import exp_types
import csv
from DataFrame.choose_experiments import choose_trajectories


class Paths(pp.Paths):
    def __init__(self, solver, size, shape, geometry, time_step=0.25):
        super().__init__()
        if 'Small' in size:
            size = 'Small'
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry
        self.time_step = time_step
        self.time_series = None
        self.single_paths = None
        self.max_subpath_length = 2

    def save_dir(self):
        name = '_'.join(['paths', self.solver, self.size, self.shape])
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_transitions')

    def load_paths(self, filenames=None):

        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

        if os.path.exists(self.save_dir() + '.txt'):
            with open(self.save_dir() + '.txt', 'r') as json_file:
                self.time_series = json.load(json_file)
                if filenames is not None:
                    self.time_series = dictfilt(self.time_series, filenames)
                self.single_paths = {name: Path(self.time_step, time_s) for name, time_s in self.time_series.items()}

        elif self.time_series is None:
            self.calculate_time_series()
            self.save_paths()
            if filenames is not None:
                self.time_series = dictfilt(self.time_series, filenames)
        [self.add_path(p) for p in self.time_series.values()]

    def calculate_time_series(self):
        if self.size == 'Small':
            size = 'Small Far'
        else:
            size = self.size
        cs_labeled = ConfigSpace_Labeled(self.solver, size, self.shape, self.geometry)
        cs_labeled.load_labeled_space()
        trajectories = choose_trajectories(solver=self.solver, size=self.size, shape=self.shape, geometry=self.geometry)
        self.single_paths = {x.filename: Path(self.time_step, conf_space_labeled=cs_labeled, x=x)
                             for x in trajectories}
        self.time_series = {name: path.time_series for name, path in self.single_paths.items()}

    def save_paths(self):
        json_string = self.time_series
        with open(self.save_dir() + '.txt', 'w') as json_file:
            json.dump(json_string, json_file)
        print('Saved paths in ', self.save_dir())


class PathsTimeStamped(Paths):
    def __init__(self, solver, size, shape, geometry, time_step=0.25):
        super().__init__(solver, size, shape, geometry)
        self.time_series = None
        self.time_stamped_series = None
        self.time_step = time_step

    def calculate_timestamped(self):
        if self.time_series is None:
            self.load_paths()
        self.time_stamped_series = {name: path.time_stamped_series() for name, path in self.single_paths.items()}

    def save_dir_timestamped(self):
        name = '_'.join([self.solver, self.size, self.shape])
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_states_timestamped')

    def save_timestamped_paths(self):
        json_string = self.time_stamped_series
        with open(self.save_dir_timestamped() + '.txt', 'w') as json_file:
            json.dump(json_string, json_file)
        print('Saved timestamped paths in ', self.save_dir_timestamped())

    def load_time_stamped_paths(self, filenames=None):
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
        if os.path.exists(self.save_dir_timestamped() + '.txt'):
            with open(self.save_dir_timestamped() + '.txt', 'r') as json_file:
                self.time_stamped_series = json.load(json_file)
                if filenames is not None:
                    self.time_stamped_series = dictfilt(self.time_stamped_series, filenames)
        else:
            print('Calculating paths')
            self.calculate_timestamped()
            self.save_timestamped_paths()
            if filenames is not None:
                self.time_stamped_series = dictfilt(self.time_stamped_series, filenames)

    def save_csv(self):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        print('Saved timestamped paths in ', self.save_dir_timestamped() + '.csv')
        with open(self.save_dir_timestamped() + '.csv', 'w') as f:
            rows = [[i] + flatten(series) for i, series in enumerate(self.time_stamped_series.values())]
            write = csv.writer(f, delimiter=" ")
            write.writerows(rows)


class PathWithoutSelfLoops(Paths):
    def load_paths(self):
        if os.path.exists(self.save_dir() + '.txt'):
            with open(self.save_dir() + '.txt', 'r') as json_file:
                self.time_series = json.load(json_file)
                self.single_paths = {name: Path(self.time_step, time_s) for name, time_s in self.time_series.items()}

        elif self.time_series is None:
            self.calculate_time_series()
            self.save_paths()
        [self.add_path(p.state_series) for p in self.single_paths.values()]


if __name__ == '__main__':
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    for size in exp_types[shape][solver]:
        paths = PathsTimeStamped(solver, size, shape, geometry)
        paths.load_paths()
        paths.load_time_stamped_paths()
        # paths.save_csv()

        # filename = list(paths.time_series.keys())[0]
        # x = get(filename)
        # x.play(path=paths.single_paths[filename], videowriter=True, step=2)
        # DEBUG = 1
