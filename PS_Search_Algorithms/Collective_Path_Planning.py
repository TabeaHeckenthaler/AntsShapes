from copy import copy
import numpy as np
from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_Maze, Node3D
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation


class Solver:
    def __init__(self, resolution):
        self.resolution = resolution
        self.distances = np.array([])


class Collective_Path_Planning(Path_planning_in_Maze):
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
    Maybe its better to think about whether points are connected via a straight line? Or calculate_diffusion_time like that?
    """

    def __init__(self, x: Trajectory_ps_simulation, starting_point: tuple, ending_point: tuple, initial_cond: str,
                 max_iter: int = 100000,
                 number_of_solvers: int = 2):
        super().__init__(x, starting_point, ending_point, initial_cond, max_iter)
        self.solvers = [Solver(None) for _ in range(number_of_solvers)]
        self.current_solver = self.choose_solver()

    def choose_solver(self) -> Solver:
        return np.random.choice(self.solvers)

    def add_knowledge(self, central_node: Node3D):
        """
        Update the low resolution node that was falsely connected (in all solvers)
        :param central_node:
        :return:
        """
        self.choose_solver()
        pass

    def warp_conf_space(self) -> np.array:
        """
        Depending on the resolution of the solvers, a planning_space for every solver is initialized.
        :return:
        """
        pass


def run_collective(shape: str, size: str, solver: str, dilation_radius: int = 8, sensing_radius: int = 7,
                   filename: str = None, show_animation: bool = False, starting_point: tuple = None,
                   ending_point: tuple = None, geometry: tuple = None, number_of_solvers=2) \
        -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param number_of_solvers: Number of solvers, which will have different resolutions.
    :param shape: shape of the load, that moves through the maze
    :param size: size of the load
    :param solver: type of solver
    :param sensing_radius: radius around the point of impact, where the planning_space is updated by the real
    conf_space
    :param dilation_radius: radius by which the conf_space is dilated by to calculate planning_space.
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
    d_star_lite = Collective_Path_Planning(x, starting_point=starting_point, ending_point=ending_point,
                                           initial_cond='back', number_of_solvers=number_of_solvers)
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    x = run_collective('SPT', 'Small Far', 'ps_simulation', dilation_radius=0, sensing_radius=100, number_of_solvers=5)
    x.play(wait=200)
    x.save()
