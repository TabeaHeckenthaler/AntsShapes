from Analysis.States import States
import os
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
import pathpy as pp
from Directories import network_dir
import json
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get


def get_trajectories(solver='human', size='Large', shape='SPT',
                     geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'), number: int = None) \
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
    df = myDataFrame[
        (myDataFrame['size'] == size) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver) &
        (myDataFrame['initial condition'] == 'back') &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]


class Paths(pp.Paths):
    def __init__(self, solver, size, shape, geometry):
        super().__init__()
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry

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
            calculated_paths = self.calculate_paths()
            for p in calculated_paths:
                self.add_path(p)
            self.save_paths()

    def calculate_paths(self):
        conf_space_labeled = ConfigSpace_Labeled(self.solver, self.size, shape, self.geometry)
        conf_space_labeled.load_labeled_space()
        trajectories = get_trajectories(solver=solver, size=size, shape=shape, geometry=geometry)
        list_of_states = [States(conf_space_labeled, x, step=int(x.fps)) for x in trajectories]
        return [s.combine_transitions(s.state_series) for s in list_of_states]

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
        print('Saved paths in ', )
