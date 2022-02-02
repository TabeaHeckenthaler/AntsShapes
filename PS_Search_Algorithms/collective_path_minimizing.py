from copy import copy
import numpy as np
from PS_Search_Algorithms.D_star_lite import D_star_lite
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation


class Collective_Path_Planning(D_star_lite):
    """
    We have multiple solvers, meaning multiple sets of resolution?
    Two ways to distort the CS:
    1. Dilation
    2. Decreasing resolution (threshold like 0.5 (?) for a pixel to be a valid configuration)
    We move according to a single solver.
    Once we encounter a wall, we add knowledge to all solvers with their given locality radius (sensing radius).
    Then we choose a new leader (randomly), which leads the group according to his minimal path.
    """
    def __init__(self, x: Trajectory_ps_simulation, sensing_radius: int, dilation_radius: int, starting_point: tuple,
                 ending_point: tuple, max_iter: int = 100000, number_of_solvers: int = 2):
        super().__init__(x, sensing_radius, dilation_radius, starting_point, ending_point, max_iter)
        self.number_of_solvers = number_of_solvers

    def path_planning(self, display_cs=True) -> None:
        """
        While the current node is not the end_screen node, and we have iterated more than max_iter
        compute the distances to the end_screen node (of adjacent nodes).
        If distance to the end_screen node is inf, break the loop (there is no solution).
        If distance to end_screen node is finite, find node connected to the
        current node with the minimal distance (+cost) (next_node).
        If you are able to walk to next_node is, make next_node your current_node.
        Else, recompute your distances.
        :param display_cs: Whether the path should be displayed during run time.
        """
        self.compute_distances()
        # self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        # self.draw_conf_space_and_path(self.known_conf_space, 'known_conf_space_fig')

        ii = 0
        while self.current.ind() != self.end.ind() and ii < self.max_iter:
            ii += 1
            if display_cs:
                self.current.draw_node(fig=self.conf_space.fig, scale_factor=0.2, color=(1, 0, 0))
            if self.current.distance == np.inf:
                return

            greedy_node = self.find_greedy_node()
            if not self.collision(greedy_node):
                greedy_node.parent = copy(self.current)
                self.current = greedy_node
            else:
                self.add_knowledge(greedy_node)
                self.compute_distances()

        if self.current.ind() == self.end.ind():
            self.winner = True
