from copy import copy
import os
import numpy as np
from skfmm import travel_time
from typing import Union

from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node3D, Node2D, Node_constructors
from ConfigSpace.ConfigSpace_Maze import ConfigSpace
import pandas as pd
from Directories import home


class Binned_ConfigSpace(ConfigSpace):
    def __init__(self, space: np.array, resolution: int):
        super().__init__(space=space)
        self.resolution = resolution
        self.binned_space = self.calculate_binned_space()
        self.node_constructor = Node_constructors[space.ndim]

    def random_ind_in_bin(self, bin_indices: tuple) -> tuple:
        """

        :param bin_indices: indices in binned space
        :return: indices in space, that are chosen from all the space_indices in the given bin
        """
        random = tuple(np.random.choice(range(self.resolution)) for _ in range(self.space.ndim))
        top_left = self.bin_ind_to_space_ind(bin_indices)
        space_indices = tuple(sum(x) for x in zip(top_left, random))
        return space_indices

    def calculate_binned_space(self) -> np.array:
        """
        :return: np.array that has dimensions self.space.shape/self.resolution, which says, how much percent coverage
        the binned space has.
        """
        # TODO
        # binned_space = ...
        # return binned_space
        return binned_space

    def space_ind_to_bin_ind(self, space_ind: tuple) -> tuple:
        return tuple(int(ind / self.resolution) for ind in space_ind)

    def bin_ind_to_space_ind(self, bin_ind: tuple) -> tuple:
        return tuple(int(ind * self.resolution) for ind in bin_ind)

    def bin_cut_out(self, indices: list) -> tuple:
        """
        Return only the cut out part where all the indices lie in. The indices have to be in adjacent bins.
        :param indices: indices in self.space
        :return: tuple of position of top, left index of cut_out in original cs, and actual cut_out
        """
        # TODO
        # return tuple, np.array([])
        return (0, 0), self.space[0:2, 0:4]

    def find_path(self, start: tuple, end: tuple) -> tuple:
        """

        :param start: indices of first node (in self.known_config_space.binned_space)
        :param end: indices of second node (in self.known_config_space.binned_space)
        :return: (boolean (whether a path was found), list (node indices that connect the two indices))
        """
        top_left, bins = self.bin_cut_out([start, end])
        Planner = Path_planning_in_CS(self.node_constructor(*start, ConfigSpace(bins)),
                                      self.node_constructor(*end, ConfigSpace(bins)),
                                      conf_space=ConfigSpace(bins))
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
        self.warp_known_conf_space()
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
        self.known_conf_space = Binned_ConfigSpace(self.known_conf_space.space, self.resolution)

    def initialize_speed(self) -> np.array:
        return Binned_ConfigSpace(self.conf_space.space, self.resolution).binned_space

    def list_distances(self, connected):
        return [self.known_conf_space.space_ind_to_bin_ind(c) for c in connected]

    def possible_step(self, greedy_node: Union[Node2D, Node3D]) -> Union[bool]:
        """
        Check if walking from self.current to greedy_node is possible.
        :param greedy_node:
        :return:
        """
        specific_node_in_bin = self.known_conf_space.random_ind_in_bin(greedy_node.ind())

        path_exists, self.found_path = self.known_conf_space.find_path(self.current.ind(), specific_node_in_bin, )
        if path_exists:
            bin_index = self.known_conf_space.space_ind_to_bin_ind(greedy_node.ind())
            manage_to_pass = np.random.uniform() > self.known_conf_space.binned_space[bin_index]
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
        self.speed = self.speed

    def compute_distances(self) -> None:
        """
        Computes travel time ( = self.distance) of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s and 0s. From the 0 line the distance metric will be calculated.
        phi = np.ones_like(self.known_conf_space.binned_space, dtype=int)
        phi[self.known_conf_space.space_ind_to_bin_ind(self.end.ind())] = 0

        print('Recompute travel time')
        self.distance = travel_time(phi, self.speed, periodic=self.periodic)

        # TO CHECK: how to plot your results in 2D in a certain plane
        # plot_distances(self, index=self.current.xi, plane='x')

    def current_known(self):
        print(self.node_constructor)
        return self.node_constructor(*self.known_conf_space.space_ind_to_bin_ind(self.current.ind()),
                                     ConfigSpace(self.known_conf_space.binned_space))

    def define_node_to_walk_to(self, ind, *args):
        return self.node_constructor(*ind, self.known_conf_space, *args)


# this is only for testing
directory = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
resolution = 2
binned_space = pd.read_excel(io=directory, sheet_name='binned_space').to_numpy()
config_space = ConfigSpace(space=pd.read_excel(io=directory, sheet_name='space').to_numpy())

if __name__ == '__main__':
    Planner = Path_Planning_Rotation_students(conf_space=config_space,
                                              start=Node2D(1, 1, config_space),
                                              end=Node2D(7, 5, config_space),
                                              resolution=resolution)

    # TODO: Draw node for Node2D and make a visualisation, so that display_cs can be made
    Planner.path_planning()
