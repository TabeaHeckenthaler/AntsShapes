from copy import copy
import numpy as np
from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node_ind
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from Directories import SaverDirectories
import os


class Path_planning_full_knowledge(Path_planning_in_CS):
    def __init__(self, x: Trajectory_ps_simulation, starting_point: tuple, ending_point: tuple, initial_cond: str,
                 max_iter: int = 100000):
        super().__init__(x, starting_point, ending_point, initial_cond, max_iter)
        x.filename = self.choose_filename(x)

    @staticmethod
    def choose_filename(x: Trajectory_ps_simulation) -> str:
        geometry_string = "__".join([string[:-5] for string in x.geometry()])
        filename = 'minimal_' + x.size + '_' + x.shape + '_' + geometry_string
        if filename in os.listdir(SaverDirectories['ps_simulation']):
            raise Exception('You are calculating a trajectory, which you already have saved')
        return filename

    def initialize_known_conf_space(self) -> np.array:
        return copy(self.conf_space)

    def add_knowledge(self, central_node: Node_ind) -> None:
        raise Exception('You should not have to add any information. Something went wrong.')


def run_full_knowledge(shape: str, size: str, solver: str, geometry: tuple, starting_point: tuple = None,
                       ending_point: tuple = None, show_animation: bool = False) \
        -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param shape: shape of the load, that moves through the maze
    :param size: size of the load
    :param solver: type of solver
    :param geometry: geometry of the maze
    :param starting_point: point (in real world coordinates) that the solver starts at
    :param ending_point: point (in real world coordinates) that the solver is aiming to reach
    :param show_animation: show animation
    :return: trajectory object.
    """
    x = Trajectory_ps_simulation(size=size, shape=shape, solver=solver, geometry=geometry)
    d_star_lite = Path_planning_full_knowledge(x, starting_point=starting_point, ending_point=ending_point)
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    x = run_full_knowledge(size='Small Far', shape='SPT', solver='ps_simulation',
                           geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
    x.play(wait=200)
    x.save()
