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
        :return: np.array that has dimensions self.space.shape/self.resolution, which says, how much percent coverage
        the binned space has.
        """
        # TODO
        binned_space = ...
        return binned_space

    def space_ind_to_bin_ind(self, space_ind: tuple) -> tuple:
        # TODO
        bin_ind = ...
        return bin_ind

    def bin_ind_to_space_ind(self, bin_ind: tuple) -> tuple:
        # TODO
        space_ind = ...
        return space_ind

    def bin_cut_out(self, indices: list) -> tuple:
        """
        Return only the cut out part where all the indices lie in. The indices have to be in adjacent bins.
        :param indices: indices in self.space
        :return: tuple of position of top, left index of cut_out in original cs, and actual cut_out
        """
        # TODO
        return tuple, np.array([])

    def find_path(self, ind1: tuple, ind2: tuple) -> tuple:
        """

        :param ind1: indices of first node (in high_resolution_space)
        :param ind2: indices of second node (in high_resolution_space)
        :return: (boolean (whether a path was found), list (node indices that connect the two indices))
        """
        top_left_ind, bins = self.bin_cut_out([ind1, ind2])
        ind1_in_bin = [sum(x) for x in zip(ind1, top_left_ind)]
        ind2_in_bin = [sum(x) for x in zip(ind2, top_left_ind)]
        Planner = Path_planning_in_CS(Node2D(*ind1_in_bin, bins),
                                      Node2D(*ind2_in_bin, bins),
                                      conf_space=bins)
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
        self.found_path = None  # saves paths, so no need to recalculate

    def step_to(self, greedy_node) -> None:
        greedy_node.parent = copy([self.found_path])
        self.found_path = None
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

    def possible_step(self, greedy_node: Union[Node2D, Node3D]) -> Union[bool]:
        path_exists, self.found_path = self.known_conf_space.find_path(self.current.ind(), greedy_node.ind())
        if path_exists:
            bin_index = self.known_conf_space.space_ind_to_bin_ind(self, greedy_node.ind())
            manage_to_pass = np.random.uniform() > self.known_conf_space.coverage()[bin_index]
            if manage_to_pass:
                return True
        else:
            return False

    def add_knowledge(self, central_node: Union[Node2D, Node3D]) -> None:
        """
        No path was found in greedy node, so we need to update our self.speed.
        Some kind of Bayesian estimation.
        :param central_node: node in high resolution config_space
        """
        # TODO
        self.speed = ...

    def compute_distances(self) -> None:
        """
        Computes travel time ( = self.distance) of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s and 0s. From the 0 line the distance metric will be calculated.
        phi = np.ones_like(self.known_conf_space.space, dtype=int)
        phi.data[self.known_conf_space.space_ind_to_bin_ind(self.end.ind())] = 0

        print('Recompute travel time')
        self.distance = travel_time(phi, self.speed, periodic=(0, 0, 1)).data

        # TO CHECK: how to plot your results in 2D in a certain plane
        # plot_distances(self, index=self.current.xi, plane='x')


if __name__ == '__main__':
    directory = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
    conf_space = pd.read_excel(io=directory, sheet_name='space').to_numpy()
    resolution = 2

    Planner = Path_Planning_Rotation_students(conf_space=ConfigSpace(space=conf_space),
                                              start=Node2D(1, 1, conf_space),
                                              end=Node2D(8, 8, conf_space),
                                              resolution=resolution)

    Planner.path_planning()
