import numpy as np
from Setup.MazeFunctions import MeasureDistance, ConnectAngle
from Setup.Attempts import Attempts
from Analysis.resolution import resolution
from copy import copy
from trajectory_inheritance.trajectory import get
from matplotlib import pyplot as plt

from Setup.Maze import Maze

# --- from experimental data--- #
# StartedScripts: check the noises (humans!!!)
noise_xy_ants_ImageAnaylsis = [0.01, 0.05, 0.02]  # cm
noise_angle_ants_ImageAnaylsis = [0.01, 0.01, 0.02]  # rad

noise_xy_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # m
noise_angle_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # rad

resolution_xy_of_ps = [0.08, 0.045]  # cm
resolution_angle_of_ps = [0.05, 0.05]  # cm


class PathLength:
    def __init__(self, x):
        self.x = copy(x)

    def during_attempts(self, *args, attempts=None, **kwargs):
        """
        Path length is calculated during attempts.
        End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
        """
        total = 0

        if attempts is None:
            attempts = Attempts(x, 'extend', *args, **kwargs)

        for attempt in attempts:
            total += self.calculate_path_length(start=attempt[0], end=attempt[1])
        # if total < Maze(size=x.size, shape=x.shape, solver=x.solver).minimal_path_length:
        #     print(x)
        #     print(total)
        #     print(x.filename + ' was smaller than the minimal path length... ')
        return total

    def per_experiment(self, plot=False, **kwargs):
        """
        Path length is calculated from beginning to end_screen.
        End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
        """
        if 'minutes' in kwargs:
            end = min(self.x.fps * kwargs['minutes'] * 60, x.frames.shape[0])
            self.x.position, self.x.angle = self.x.position[0: end], self.x.angle[0: end]

        return self.calculate_path_length(plot=plot)

    def calculate_path_length(self, start=0, end=-1, rot=True, plot=False):
        """
        Reduce path to a list of points that each have distance of at least resolution = 0.1cm
        to the next point.
        Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
        Path length the sum of the distances of the points in the list.
        """
        position, angle = self.x.position[start: end], self.x.angle[start: end]
        aver_radius = Maze(self.x).average_radius()

        unwrapped_angle = ConnectAngle(angle[1:], self.x.shape)
        pos_list, ang_list = [], []

        if unwrapped_angle.size == 0 or position.size == 0:
            return 0
        pos, ang = position[0], unwrapped_angle[0]
        path_length = 0
        for i in range(len(unwrapped_angle)):
            d = MeasureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
            if d < resolution(self.x.size, self.x.solver):
                pass
            else:
                path_length += MeasureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
                pos, ang = position[i], unwrapped_angle[i]
                if plot:
                    pos_list.append(pos)
                    ang_list.append(ang)
        if plot:
            plt.plot(position[:, 0], position[:, 1], color='blue')
            plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1], color='k')
            plt.show()

        return path_length

    def minimal(self):
        if self.x.shape not in ['RASH', 'LASH']:
            ideal_filename = 'XL_' + self.x.shape + '_dil0_sensing1'
            ideal = get(ideal_filename)
            return PathLength(ideal).per_experiment() * Maze(self.x).exit_size/Maze(ideal).exit_size
        else:
            return np.nan


if __name__ == '__main__':
    from trajectory_inheritance.trajectory import get, sizes

    # p = [resolution(size, 'ant') for size in sizes['ant']]
    x = get('medium_20210901010920_20210901011020_20210901011020_20210901011022_20210901011022_20210901011433')
    # x = Get('XL_SPT_4290008_XLSpecialT_1_ants', 'ant')
    print(PathLength(x).per_experiment(plot=True))
