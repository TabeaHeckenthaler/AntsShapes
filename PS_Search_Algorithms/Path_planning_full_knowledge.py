from copy import copy
import numpy as np
from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_Maze, Node3D
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from Directories import SaverDirectories
import os


def minimal_filename(size: str, shape: str, geometry: tuple, initial_cond) -> str:
    if initial_cond is None:
        initial_cond = ''
    if len(initial_cond) > 0:
        initial_cond = '_' + initial_cond
    geometry_string = "_".join([string[:-5] for string in geometry])
    return 'minimal_' + size + '_' + shape + initial_cond + '_' + geometry_string


class Path_planning_full_knowledge(Path_planning_in_Maze):
    def __init__(self, x: Trajectory_ps_simulation, starting_node: Node3D, ending_node: Node3D,
                 initial_cond: str = '', max_iter: int = 100000):
        super().__init__(x, starting_node, ending_node, initial_cond, max_iter)
        x.filename = self.choose_filename(x, initial_cond)

    @staticmethod
    def choose_filename(x: Trajectory_ps_simulation, initial_cond: str = '') -> str:
        filename = minimal_filename(x.size, x.shape, x.geometry(), initial_cond)
        if filename in os.listdir(SaverDirectories['ps_simulation']):
            print('You are calculating a trajectory, which you already have saved:' + filename)
        return filename

    def warp_conf_space(self) -> np.array:
        return copy(self.conf_space)

    def add_knowledge(self, central_node: Node3D) -> None:
        raise Exception('You should not have to add any information. Something went wrong.')


def connector(part1, part2, frames_missing, filename=None):
    if filename is None:
        filename = part1.VideoChain[-1] + '_CONNECTOR_' + part2.filename

    connector_load = run_full_knowledge(shape=part1.shape, size=part1.size, solver=part1.solver,
                                        starting_point=(part1.position[-1][0], part1.position[-1][1], part1.angle[-1]),
                                        ending_point=(part2.position[0][0], part2.position[0][1], part2.angle[0]),
                                        geometry=part1.geometry())
    connector_load.filename = filename
    connector_load.stretch(frames_missing)
    connector_load.tracked_frames = [connector_load.frames[0], connector_load.frames[-1]]
    connector_load.falseTracking = []
    connector_load.free = part1.free
    return connector_load


def run_full_knowledge(shape: str, size: str, solver: str, geometry: tuple, starting_node: Node3D = None,
                       ending_node: Node3D = None, initial_cond: str = '', show_animation: bool = False) \
        -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param initial_cond:
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
    d_star_lite = Path_planning_full_knowledge(x, starting_node=starting_node, ending_node=ending_node,
                                               initial_cond=initial_cond)
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    for size in ['S']:
        x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
                               geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                               initial_cond='back')
        x.save()

    # for size in ['XL', 'L', 'M', 'S']:
    #     x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
    #                            geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
    #                            initial_cond='front')
    #     x.save()
    #
    # for size in ['Small Far', 'Medium', 'Large']:
    #     x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
    #                            geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
    #                            initial_cond='front')
    #     x.save()
    #
    # for size in ['Small Far', 'Medium', 'Large']:
    #     x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
    #                            geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
    #                            initial_cond='back')
    #     x.save()

