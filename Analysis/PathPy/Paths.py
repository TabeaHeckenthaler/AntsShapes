from Analysis.PathPy.Path import Path
import os
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
import pathpy as pp
from Directories import network_dir
import json
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get
from trajectory_inheritance.exp_types import exp_types


def get_trajectories(solver='human', size='Large', shape='SPT',
                     geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                     number: int = None) \
        -> list:
    """
    Get trajectories based on the trajectories saved in myDataFrame.
    :param geometry: geometry of the maze, given by the names of the excel files with the dimensions
    :param number: How many trajectories do you want to have in your list?
    :param solver: str
    :param size: str
    :param shape: str
    :return: list of objects, that are of the class or subclass Trajectory
    """
    if size == 'Small':
        sizes = ['Small Far', 'Small Near']
    else:
        sizes = [size]
    df = myDataFrame[
        (myDataFrame['size'].isin(sizes)) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver) &
        (myDataFrame['initial condition'] == 'back') &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]


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

    def save_dir(self):
        name = '_'.join(['network', self.solver, self.size, self.shape])
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_transitions.txt')

    def add_paths(self, transitions: list) -> None:
        [self.add_path(transition) for transition in transitions]

    def load_paths(self):
        if os.path.exists(self.save_dir()):
            with open(self.save_dir(), 'r') as json_file:
                saved_paths = json.load(json_file)
            [self.add_path(p) for p in saved_paths]
        else:
            if self.time_series is None:
                self.calculate_time_series()
            for p in self.time_series:
                self.add_path(p)
            self.save_paths()

    def calculate_time_series(self):
        if self.size == 'Small':
            size = 'Small Far'
        else:
            size = self.size
        cs_labeled = ConfigSpace_Labeled(self.solver, size, self.shape, self.geometry)
        cs_labeled.load_labeled_space()
        trajectories = get_trajectories(solver=self.solver, size=self.size, shape=self.shape, geometry=self.geometry)
        self.paths = {x.filename: Path(cs_labeled, x, step=int(x.fps * self.time_step)) for x in trajectories}
        self.time_series = {name: path.time_series for name, path in self.paths.items()}

    def to_list(self):
        lister = []
        for p_length in self.paths:
            for p in self.paths[p_length]:
                segment = []
                for s in p:
                    segment.append(s)
                for _ in range(int(self.paths[p_length][p][1])):
                    lister += [list(segment)]
        return lister

    def save_paths(self):
        json_string = self.to_list()
        with open(self.save_dir(), 'w') as json_file:
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
            self.calculate_time_series()
        self.time_stamped_series = {name: path.time_stamped_series() for name, path in self.paths.items()}

    def save_dir(self):
        name = '_'.join(['network', self.solver, self.size, self.shape])
        return os.path.join(network_dir, 'ExperimentalPaths', name + '_states_timestamped.txt')

    def to_list(self):
        return self.time_stamped_series

    def load_paths(self):
        if os.path.exists(self.save_dir()):
            with open(self.save_dir(), 'r') as json_file:
                self.time_stamped_series = json.load(json_file)
        else:
            print('Calculating paths')
            self.calculate_timestamped()


if __name__ == '__main__':
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    for size in exp_types[shape][solver][-1:]:
        paths = PathsTimeStamped(solver, size, shape, geometry)
        paths.load_paths()

        filename = list(paths.paths.keys())[0]
        x = get(filename)
        x.play(path=paths.paths[filename], videowriter=True)
        paths.save_paths()
        # TODO
