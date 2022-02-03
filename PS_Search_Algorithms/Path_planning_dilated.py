from copy import copy
import numpy as np
from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node_ind, structure
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from scipy.ndimage.measurements import label
from Directories import SaverDirectories
import os


class Path_planning_dilated(Path_planning_in_CS):
    def __init__(self, x: Trajectory_ps_simulation, starting_point: tuple, ending_point: tuple,
                 sensing_radius: int, dilation_radius: int, initial_cond: str, max_iter: int = 100000):
        super().__init__(x, starting_point, ending_point, initial_cond, max_iter)
        self.dilation_radius = dilation_radius
        self.sensing_radius = sensing_radius
        x.filename = self.choose_filename(x)

    def choose_filename(self, x: Trajectory_ps_simulation) -> str:
        filename = x.size + '_' + x.shape + '_' + 'dil' + str(self.dilation_radius) + '_sensing' + \
                   str(self.sensing_radius)
        if filename in os.listdir(SaverDirectories['ps_simulation']):
            raise Exception('You are calculating a trajectory, which you already have saved')
        return filename

    def initialize_known_conf_space(self) -> np.array:
        known_conf_space = copy(self.conf_space)
        if self.dilation_radius > 0:
            known_conf_space = known_conf_space.dilate(space=self.conf_space.space, radius=self.dilation_radius)
        return known_conf_space

    def add_knowledge(self, central_node: Node_ind) -> None:
        """
        Adds knowledge to the known configuration space of the solver with a certain sensing_radius around
        the central node, which is the point of interception
        :param central_node: point of impact, which is the center of where the maze will be updated
        """
        # roll the array
        rolling_indices = [- max(central_node.xi - self.sensing_radius, 0),
                           - max(central_node.yi - self.sensing_radius, 0),
                           - (central_node.thetai - self.sensing_radius)]

        conf_space_rolled = np.roll(self.conf_space.space, rolling_indices, axis=(0, 1, 2))
        known_conf_space_rolled = np.roll(self.known_conf_space.space, rolling_indices, axis=(0, 1, 2))

        # only the connected component which we sense
        sr = self.sensing_radius
        labeled, _ = label(conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], structure)
        known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr] = \
            np.logical_or(
                np.array(known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], dtype=bool),
                np.array(labeled == labeled[sr, sr, sr])).astype(int)

        # update_screen known_conf_space by using known_conf_space_rolled and rolling back
        self.known_conf_space.space = np.roll(known_conf_space_rolled, [-r for r in rolling_indices], axis=(0, 1, 2))


def run_dilated(shape: str, size: str, solver: str, filename: str = None, show_animation: bool = False,
                starting_point: tuple = None, ending_point: tuple = None, geometry: tuple = None,
                dilation_radius: int = 0, sensing_radius: int = 100) \
        -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param sensing_radius:
    :param dilation_radius:
    :param shape: shape of the load, that moves through the maze
    :param size: size of the load
    :param solver: type of solver
    :param filename: Name of the trajectory
    :param show_animation: show animation
    :param starting_point: point (in real world coordinates) that the solver starts at
    :param ending_point: point (in real world coordinates) that the solver is aiming to reach
    :param geometry: geometry of the maze
    :return: trajectory object.
    """
    x = Trajectory_ps_simulation(size=size, shape=shape, solver=solver, filename=filename, geometry=geometry)
    d_star_lite = Path_planning_dilated(x, starting_point=starting_point, ending_point=ending_point,
                                        sensing_radius=sensing_radius, dilation_radius=dilation_radius, init_cond='back'
                                        )
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    x = run_dilated('SPT', 'Small Far', 'ps_simulation', dilation_radius=0, sensing_radius=100)
    x.play(wait=200)
    x.save()

    # TODO: WAYS TO MAKE LESS EFFICIENT:
    #  limited memory
    #  locality (patch size)
    #  accuracy of greedy node, add stochastic behaviour
    #  false walls because of limited resolution

    # === For parallel processing multiple trajectories on multiple cores of your computer ===
    # Parallel(n_jobs=6)(delayed(run_dstar)(sensing_radius, dil_radius, shape)
    #                    for dil_radius, sensing_radius, shape in
    #                    itertools.product(range(0, 16, 1), range(1, 16, 1), ['SPT'])
    #                    # itertools.product([0], [0], ['H', 'I', 'T'])
    #                    )
