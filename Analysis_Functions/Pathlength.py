import numpy as np
from Setup.MazeFunctions import MeasureDistance, ConnectAngle
from Setup.Attempts import Attempts
from Setup.Maze import Maze, ResizeFactors
from Setup.Load import average_radius
from Analysis_Functions.resolution import resolution
from matplotlib import pyplot as plt
from progressbar import progressbar

# --- from experimental data--- #
# TODO: check the noises (humans!!!)
noise_xy_ants_ImageAnaylsis = [0.01, 0.05, 0.02]  # cm
noise_angle_ants_ImageAnaylsis = [0.01, 0.01, 0.02]  # rad

noise_xy_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # m
noise_angle_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # rad

resolution_xy_of_ps = [0.08, 0.045]  # cm
resolution_angle_of_ps = [0.05, 0.05]  # cm


def calculate_path_length(position, angle, aver_radius, shape, size, solver, rot=True, plot=False):
    """
    Reduce path to a list of points that each have distance of at least resolution = 0.1cm
    to the next point.
    Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
    Path length the sum of the distances of the points in the list.
    """

    unwrapped_angle = ConnectAngle(angle[1:], shape)
    pos_list, ang_list = [], []
    if unwrapped_angle.size == 0 or position.size == 0:
        return 0
    pos, ang = position[0], unwrapped_angle[0]
    path_length = 0
    for i in progressbar(range(len(unwrapped_angle))):
        d = MeasureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
        if d < resolution(size, solver):
            pass
        else:
            path_length += MeasureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
            pos, ang = position[i], unwrapped_angle[i]
            if plot:
                pos_list.append(pos)
                ang_list.append(ang)
    plt.plot(position[:, 0], position[:, 1], color='blue')
    plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1], color='k')
    plt.show()

    return path_length


from trajectory import Get, sizes
p = [resolution(size, 'ant') for size in sizes['ant']]
x = Get('M_H_4180002_1_ants', 'ant')
# x = Get('XL_SPT_4290008_XLSpecialT_1_ants', 'ant')
calculate_path_length(x.position, x.angle, average_radius(x.size, x.shape, x.solver), x.shape, x.size, x.solver,
                      rot=True, plot=True)


def path_length_per_experiment(x, **kwargs):
    """
    Path length is calculated from beginning to end.
    End is either given through the kwarg 'minutes', or is defined as the end of the experiment.
    """
    if 'minutes' in kwargs:
        end = min(x.fps * kwargs['minutes'] * 60, x.frames.shape[0])
    else:
        end = len(x.angle)

    return calculate_path_length(x.position[0: end],
                                 x.angle[0: end],
                                 average_radius(x.size, x.shape, x.solver),
                                 x.shape,
                                 x.size,
                                 x.solver,
                                 rot=True)


def path_length_during_attempts(x, *args, attempts=None, **kwargs):
    """
    Path length is calculated during attempts.
    End is either given through the kwarg 'minutes', or is defined as the end of the experiment.
    """
    total = 0

    if attempts is None:
        attempts = Attempts(x, 'extend', *args, **kwargs)

    for attempt in attempts:
        total += calculate_path_length(x.position[attempt[0]: attempt[1]],
                                       x.angle[attempt[0]: attempt[1]],
                                       average_radius(x.size, x.shape, x.solver), x.shape, x.size, x.solver, rot=True)
    #
    # if total < Maze(size=x.size, shape=x.shape, solver=x.solver).minimal_path_length:
    #     print(x)
    #     print(total)
    #     print(x.filename + ' was smaller than the minimal path length... ')

    return total


def mean_path_length_per_attempt(x, *args, attempts=None, **kwargs):
    path_per_att = list()

    if attempts is None:
        attempts = Attempts(x, 'extend', *args, **kwargs)

    for attempt in attempts:
        path_per_att.append(calculate_path_length(x.position[attempt[0]: attempt[1]],
                                                  x.angle[attempt[0]: attempt[1]],
                                                  average_radius(x.size, x.shape, x.solver),
                                                  x.shape, x.size, x.solver, rot=True,
                                                  )
                            )
    return np.mean(path_per_att)
