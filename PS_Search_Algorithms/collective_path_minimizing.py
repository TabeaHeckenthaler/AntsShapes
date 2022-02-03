from copy import copy
import numpy as np
from PS_Search_Algorithms.D_star_lite import D_star_lite
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation


class Collective_Path_Planning(D_star_lite):
    """
    Differences from D_star_lite
    Collective =
    Multiple solvers, which each have a different resolution. # TODO not sure how to chose the resolutions.

    Distortion of CS in known_phase_space =
    1. Dilation
    2. Decreasing resolution (A set of pixels are combined to a big-pixel => resolution).
    A set of pixels constitute a big pixel. If a valid configuration is found in the set of pixels,
    the big pixel as a whole is also valid. This way, we will connect areas that beforehand were not connected.

    Path Planning =
    At the beginning, a solver is chosen randomly as the 'group leader', and his distance matrix is calculated on his
    low resolution representation of the maze.
    The solver finds node out of the set of neighboring nodes to his current node, which minimizes the distance to
    the goal node.
    We check whether the step is possible, by extracting the high resolution sets of pixels, and seeing wether there is
    a path from one to the other, only by using these sets of pixels. # TODO not sure how to do this computationally
    If a step is being chosen, which is impossible (on the high resolution CS), we add this knowledge to all solvers
    by setting the appropriate connecting pixels to 'False'.  # TODO (not so clear yet)
    Then we choose a new solver (randomly), which leads the group according to his minimal path.

    Notes =
    Computation time will stay lower, if we calculate distances on the low resolution CS.

    Question =
    If we only had two big pixels, the maze would be solved easily. Hence, this only works, if we have a resolution high
    enough to capture 'the intricacies' of the maze. Something like shortest distance between two states?
    I am actually not sure about this point.
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


def run_collective_path_planning(shape: str, size: str, solver: str, dilation_radius: int = 8, sensing_radius: int = 7,
                                 filename: str = None, show_animation: bool = False, starting_point: tuple = None,
                                 ending_point: tuple = None, geometry: tuple = None, number_of_solvers=2) \
        -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param number_of_solvers: Number of solvers, which will have different resolutions.
    :param shape: shape of the load, that moves through the maze
    :param size: size of the load
    :param solver: type of solver
    :param sensing_radius: radius around the point of impact, where the known_conf_space is updated by the real
    conf_space
    :param dilation_radius: radius by which the conf_space is dilated by to calculate known_conf_space.
    :param filename: Name of the trajectory
    :param show_animation: show animation
    :param starting_point: point (in real world coordinates) that the solver starts at
    :param ending_point: point (in real world coordinates) that the solver is aiming to reach
    :param geometry: geometry of the maze
    :return: trajectory object.
    """
    if filename is None:
        filename = 'test'
    # elif filename in os.listdir(SaverDirectories['ps_simulation']):
    #     return
    x = Trajectory_ps_simulation(size=size, shape=shape, solver=solver, filename=filename, geometry=geometry)
    d_star_lite = Collective_Path_Planning(x, sensing_radius=sensing_radius, dilation_radius=dilation_radius,
                                           starting_point=starting_point, ending_point=ending_point,
                                           number_of_solvers=number_of_solvers)
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    x = run_collective_path_planning('SPT', 'Small Far', 'ps_simulation', dilation_radius=0, sensing_radius=100,
                                     number_of_solvers=5)
    x.play(wait=200)
    x.save()
