from trajectory_inheritance.trajectory import get, Trajectory
import numpy as np
from Analysis.PathLength import PathLength
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from matplotlib import pyplot as plt


def resolution(size: str, shape: str, solver: str, geometry: tuple, path_length: float) -> int:
    """
    The goal is to find the proper bin size.
    :param size:
    :param shape:
    :param solver:
    :return:
    """
    conf_space = ConfigSpace_Maze(solver, size, shape, geometry)
    conf_space.load_space()
    return int(conf_space.coords_to_index(0, path_length))


def typical_v_max() -> float:
    # TODO Rotation students: Find typical top speed
    return float()
# how long does it take for the shape to reach from 0 to v_max or from v_max to 0? Better: How much path is traversed.


def find_acceleration_frames(x: Trajectory) -> list:
    # TODO Rotation student: Find for a given trajectory, example frames, where the load is accelerated from 0 to v_max,
    #   or where the shape is decelerated (far away from the wall)

    # Find frames, where v is about v_max

    # Find frames beforehand, where the speed is 0.

    pass


# saved in trajectory_inheritance/trajectories_local
exp = {'Large': ['large_20211006174255_20211006174947'],  # 26 people
       'Medium': ['medium_20211125132354_20211125133125_20211125133125_20211125133138_20211125133138_20211125134057'],  #  9 people
       'Small Far': ['small_20210413102456_20210413102801']  # 1 person
       }


if __name__ == '__main__':
    path_list = list()  # list
    shape, solver = 'SPT', 'human'
    geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')  # tuple
    resolution_dict = dict()
    for size in exp.keys():
        for name in exp[size]:
            trajectory = get(name)

            # trajectory.position
            # trajectory.angle
            # trajectory.angle.__class__ gives you the class of your object

            # TODO Itay: Plot position (x, t), plot (speed, t) (-> matplotlib.pyplot)
            plt.figure()

            # frames = find_acceleration_frames(traj)
            frames = [1, 100]
            path_list.append(PathLength(trajectory).calculate_path_length(frames=frames))

        resolution_dict[size] = resolution(size, shape, solver, geometry, np.mean(path_list))
    print(resolution_dict)

