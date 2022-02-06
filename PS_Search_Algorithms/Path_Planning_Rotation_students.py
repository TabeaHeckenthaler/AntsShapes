from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node_ind
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
import numpy as np
from skfmm import travel_time
from PhaseSpaces.PhaseSpace import PhaseSpace
from copy import copy


class Binned_PhaseSpace(PhaseSpace):
    def __init__(self, phase_space: PhaseSpace, resolution: int):
        super().__init__(solver=phase_space.solver, size=phase_space.size, shape=phase_space.shape,
                         geometry=phase_space.geometry, space=phase_space.space)
        self.resolution = resolution
        self.binned_space = self.calculate_binned_space()

    def calculate_binned_space(self) -> np.array:
        """
        :return: Space that has
        """
        return self.space

    def bins_connected(self, ind1, ind2) -> bool:
        return self.find_path(ind1, ind2)[0]

    def find_path(self, ind1, ind2) -> tuple:
        """

        :param ind1: indices of first node (in high_resolution_space)
        :param ind2: indices of second node (in high_resolution_space)
        :return: (boolean (whether a path was found), list (node indices that connect the two indices))
        """
        # TODO: check whether centers of voxels are connected.
        path_exists = False
        if not path_exists:
            return False, None
        else:
            return True, [ind1, ind2]

    def coverage(self) -> np.array:
        # TODO: calculate coverage for binned space
        return np.array([])


class Path_Planning_Rotation_students(Path_planning_in_CS):
    def __init__(self, x: Trajectory_ps_simulation, starting_point: tuple, ending_point: tuple, initial_cond: str,
                 resolution: int, max_iter: int = 100000, dil_radius: int = 0):
        super().__init__(x, starting_point, ending_point, initial_cond, max_iter)
        self.dil_radius = dil_radius
        self.speed = self.initialize_speed()
        self.resolution = resolution
        # TODO Tabea: Greedy node on high resolution conf_space!!!!

    def step_to(self, greedy_node) -> None:
        greedy_node.parent = copy([self.known_conf_space.find_path(self.current.ind(), greedy_node.ind())[1]])
        self.current = greedy_node

    def initialize_known_conf_space(self) -> np.array:
        """
        Known configuration space is a low resolution representation of the real maze.
        :return:
        """
        self.known_conf_space.space = copy(self.conf_space.space)
        if self.dil_radius > 0:
            self.known_conf_space.space = self.conf_space.dilate(self.known_conf_space.space, radius=self.dil_radius)
        self.known_conf_space.space = Binned_PhaseSpace(self.known_conf_space.space, self.resolution)

    def initialize_speed(self) -> np.array:
        return 1/Binned_PhaseSpace(self.conf_space, self.resolution).coverage()

    def possible_step(self, greedy_node: Node_ind) -> bool:
        if self.known_conf_space.bins_connected(self.current.ind(), greedy_node.ind()):
            manage_to_pass = np.random.uniform() > self.known_conf_space.coverage()[greedy_node.ind()]
            if manage_to_pass:
                return False
            else:
                return True
        else:
            return False

    def add_knowledge(self, central_node: Node_ind) -> None:
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

