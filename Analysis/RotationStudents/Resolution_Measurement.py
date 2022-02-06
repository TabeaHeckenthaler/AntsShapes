from trajectory_inheritance.trajectory import get, Trajectory, Trajectory_part
import numpy as np


def resolution(size: str, shape: str, solver: str, path_length: float) -> int:
    """
    The goal is to find the proper bin size.
    :param size:
    :param shape:
    :param solver:
    :return:
    """
    return int()


# how long does it take for the shape to reach from 0 to v_max or from v_max to 0? Better: How much path is traversed.
def find_acceleration_frames(x: Trajectory) -> list:
    return []


def path_traversed(x: Trajectory, frames: list) -> float:
    return Trajectory_part.pathlength()  # TODO: Tabea


exp = {'Large': {'name1': [], 'name2': []}}

if __name__ == '__main__':
    path_list = []
    for size in exp.keys():
        for name, frames in exp[size].items():
            traj = get(name)
            frames = find_acceleration_frames(traj)
            path_list.append(path_traversed(traj, frames))

        res = resolution(size, 'SPT', 'human', np.mean(path_list))
        print(size + ': ', str(res))
