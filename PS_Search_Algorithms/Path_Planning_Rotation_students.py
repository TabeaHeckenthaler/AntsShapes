from copy import copy
import os
import numpy as np
from skfmm import travel_time
from typing import Union

from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node3D, Node2D
from ConfigSpace.ConfigSpace_Maze import ConfigSpace
import pandas as pd
from Directories import home


class Binned_ConfigSpace(ConfigSpace):
    def __init__(self, config_space: ConfigSpace, resolution: int):
        super().__init__(space=config_space.space)
        self.resolution = resolution
        self.binned_space = self.calculate_binned_space()

    def calculate_binned_space(self) -> np.array:
        """
        :return: np.array that has dimensions self.space.shape/self.resolution, which says, how much percent coverage the
        binned space has.
        """
        # TODO
        return np.array([])

    def bin_cut_out(self, indices: list) -> np.array:
        """
        Return only the cut out part where all the indices lie in. The indices have to be in adjacent bins.
        :param indices: indices in self.space
        :return:
        """
        # TODO
        return np.array([])

    def find_path(self, ind1: tuple, ind2: tuple) -> tuple:
        """

        :param ind1: indices of first node (in high_resolution_space)
        :param ind2: indices of second node (in high_resolution_space)
        :return: (boolean (whether a path was found), list (node indices that connect the two indices))
        """
        bins = self.bin_cut_out([ind1, ind2])
        # path only within the adjacent bins
        # TODO Tabea: I think, we have to change the indices to new cut out.
        Planner = Path_planning_in_CS(Node2D(*ind1, bins), Node2D(*ind2, bins), conf_space=bins)
        Planner.path_planning()

        if Planner.winner:
            return Planner.winner, Planner.generate_path()
        else:
            return False, None


class Path_Planning_Rotation_students(Path_planning_in_CS):
    def __init__(self, conf_space, start: Union[Node2D, Node3D], end: Union[Node2D, Node3D], resolution: int,
                 max_iter: int = 100000, dil_radius: int = 0):
        super().__init__(start, end, max_iter, conf_space=conf_space)
        self.dil_radius = dil_radius
        self.resolution = resolution
        self.speed = self.initialize_speed()
        DEBUG = 1

    def step_to(self, greedy_node) -> None:
        greedy_node.parent = copy([self.known_conf_space.find_path(self.current.ind(), greedy_node.ind())[1]])
        self.current = greedy_node

    def warp_known_conf_space(self) -> np.array:
        """
        Known configuration space is a low resolution representation of the real maze.
        :return:
        """
        self.known_conf_space.space = copy(self.conf_space.space)
        if self.dil_radius > 0:
            self.known_conf_space.space = self.conf_space.dilate(self.known_conf_space.space, radius=self.dil_radius)
        self.known_conf_space.space = Binned_ConfigSpace(self.known_conf_space.space, self.resolution)

    def initialize_speed(self) -> np.array:
        return Binned_ConfigSpace(self.conf_space, self.resolution).binned_space

    def possible_step(self, greedy_node: Union[Node2D, Node3D]) -> Union[bool, list]:
        path_exists, path = self.known_conf_space.bins_connected(self.current.ind(), greedy_node.ind())
        if path_exists:
            manage_to_pass = np.random.uniform() > self.known_conf_space.coverage()[greedy_node.ind()]
            if manage_to_pass:
                return path
            else:
                return True
        else:
            return False

    def add_knowledge(self, central_node: Union[Node2D, Node3D]) -> None:
        """
        No path was found in greedy node, so we need to update our self.speed.
        Some kind of Bayesian estimation.
        :param central_node:
        :return:
        """
        # TODO: Some Bayesian estimation update, on the... speed? or Coverage? (Tabea)

    def compute_distances(self) -> None:
        """
        Computes travel time ( = self.distance) of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s and 0s. From the 0 line the distance metric will be calculated.
        phi = np.ones_like(self.known_conf_space.space, dtype=int)

        mask = ~self.known_conf_space.space
        phi = np.ma.MaskedArray(phi, mask)
        phi.data[self.end.ind()] = 0

        print('Recompute travel time')
        self.distance = travel_time(phi, self.speed, periodic=(0, 0, 1)).data
        # in order to mask the 'unreachable' nodes (diagonal or outside of conf_space), set distance there to inf.
        # dist = travel_time(phi, self.speed, periodic=(0, 0, 1))
        # dist_data = dist.data
        # dist_data[dist.mask] = np.inf
        # self.distance = dist_data

        # TO CHECK: how to plot your results in 2D in a certain plane
        # plot_distances(self, index=self.current.xi, plane='x')


if __name__ == '__main__':
    directory = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
    conf_space = pd.read_excel(io=directory, sheet_name='space').to_numpy()
    resolution = 2

    Planner = Path_Planning_Rotation_students(conf_space=ConfigSpace(space=conf_space),
                                              start=Node2D(0, 0, conf_space),
                                              end=Node2D(8, 8, conf_space),
                                              resolution=resolution)
