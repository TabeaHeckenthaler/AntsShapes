from Setup.Attempts import Attempts
from trajectory_inheritance.trajectory import Trajectory_part
import numpy as np
from Analysis.resolution import resolution
from copy import copy
from trajectory_inheritance.get import get
from matplotlib import pyplot as plt
from Setup.Maze import Maze
from PS_Search_Algorithms.Path_planning_full_knowledge import Path_planning_full_knowledge
from Analysis.average_carrier_number.averageCarrierNumber import myDataFrame
from tqdm import tqdm
from Directories import path_length_dir, penalized_path_length_dir
import json
from Setup.MazeFunctions import ConnectAngle


# --- from experimental data--- #
# StartedScripts: check the noises (humans!!!)
# noise_xy_ants_ImageAnaylsis = [0.01, 0.05, 0.02]  # cm
# noise_angle_ants_ImageAnaylsis = [0.01, 0.01, 0.02]  # rad
#
# noise_xy_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # m
# noise_angle_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # rad
#
# resolution_xy_of_ps = [0.08, 0.045]  # cm
# resolution_angle_of_ps = [0.05, 0.05]  # cm


class PathLength:
    def __init__(self, x):
        self.x = copy(x)

    def during_attempts(self, *args, attempts=None, **kwargs):
        return None
    #     # TODO
    #     """
    #     Path length is calculated during attempts.
    #     End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
    #     """
    #     return None
    #     total = 0
    #
    #     if attempts is None:
    #         attempts = Attempts(self.x, 'extend', *args, **kwargs)
    #
    #     for attempt in attempts:
    #         total += self.calculate_path_length(start=attempt[0], end=attempt[1])
    #     # if total < Maze(size=x.size, shape=x.shape, solver=x.solver).minimal_path_length:
    #     #     print(x)
    #     #     print(total)
    #     #     print(x.filename + ' was smaller than the minimal path length... ')
    #     return total

    def per_experiment(self, penalize=False) -> float:
        """
        Path length is calculated from beginning to end_screen.
        End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
        """
        # I have to split movies, because for 'connector movies', we have to treat them separately.
        if penalize and (self.x.solver != 'ant' or self.x.shape != 'SPT'):
            return np.NaN
        parts = self.x.divide_into_parts()
        path_lengths = [PathLength(part).calculate_path_length(penalize=penalize) for part in parts]
        interpolated_path_lengths = self.interpolate_connectors(parts, path_lengths)
        return np.sum(interpolated_path_lengths)

    def interpolate_connectors(self, parts, path_lengths) -> list:
        """
        :param parts: parts of trajectories
        :param path_lengths: calculated path lengths which contain nans.
        :return: path lengths without nans.
        """
        missed_frames = np.sum([len(part.frames) for part, path_length in
                                zip(parts, path_lengths) if np.isnan(path_length)])
        total_length = self.x.frames.shape[0]
        path_length_per_frame = np.nansum(path_lengths)/(total_length - missed_frames)
        return [len(part.frames) * path_length_per_frame
                if np.isnan(path_length) else path_length for part, path_length in zip(parts, path_lengths)]

    @staticmethod
    def measureDistance(position1, position2, angle1, angle2, averRad, rot=True, **kwargs):  # re`turns distance in cm.
        archlength = 0
        if position1.ndim == 1:  # For comparing only two positions
            translation = np.linalg.norm(position1[:2] - position2[:2])
            if rot:
                archlength = abs(angle1 - angle2) * averRad

        else:  # For comparing more than 2 positions
            # translation = np.sqrt(np.sum(np.power((position1[:, :2] - position2[:, :2]), 2), axis=1))
            translation = np.linalg.norm(position1[:, :2] - position2[:, :2])
            if rot:
                archlength = abs(angle1[:] - angle2[:]) * averRad
        return translation + archlength

    def average_radius(self):
        return Maze(self.x).average_radius()

    # def calculate_path_length_old(self, rot: bool = True, frames: list = None, max_path_length=np.inf):
    #     """
    #     Reduce path to a list of points that each have distance of at least resolution = 0.1cm
    #     to the next point.
    #     Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
    #     Path length the sum of the distances of the points in the list.
    #     """
    #     if frames is None:
    #         frames = [0, -1]
    #
    #     # the connector parts dont have long enough path length.
    #     if isinstance(self.x, Trajectory_part) and self.x.is_connector():
    #         raise ValueError('Check here')
    #
    #     position, angle = self.x.position[frames[0]: frames[1]], self.x.angle[frames[0]: frames[1]]
    #     aver_radius = self.average_radius()
    #
    #     unwrapped_angle = ConnectAngle(angle[1:], self.x.shape)
    #     if unwrapped_angle.size == 0 or position.size == 0:
    #         return 0
    #     pos, ang = position[0], unwrapped_angle[0]
    #     path_length = 0
    #     cs_resolution = resolution(self.x.geometry(), self.x.size, self.x.solver, self.x.shape)
    #
    #     for i in range(1, len(unwrapped_angle)):
    #         d = self.measureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
    #         if d > cs_resolution:
    #             path_length += self.measureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
    #             if path_length > max_path_length:
    #                 return path_length
    #             pos, ang = position[i], unwrapped_angle[i]
    #     return path_length

    def calculate_path_length(self, rot: bool = True, frames: list = None, penalize=False, kernel_size=None,
                              max_path_length=np.inf):
        """
        Reduce path to a list of points that each have distance of at least resolution = 0.1cm
        to the next point.
        Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
        Path length the sum of the distances of the points in the list.
        When the shape is standing still, the path length increases. Penalizing for being stuck.
        """
        print(self.x.filename)
        if frames is None:
            frames = [0, -1]
        position, angle = self.x.position[frames[0]: frames[1]], self.x.angle[frames[0]: frames[1]]

        if kernel_size is None:
            kernel_size = 2 * (self.x.fps // 4) + 1
        position_filtered, unwrapped_angle_filtered = self.x.smoothed_pos_angle(position, angle, kernel_size)
        stuck_frames = (np.zeros(angle.size)).astype(bool)

        DEBUG = 1
        if penalize and \
                self.x.size not in ['S', 'XS'] and \
                myDataFrame[myDataFrame['filename'] == self.x.filename].iloc[0]['average Carrier Number'] > 1:

            # Calculate velocity in units of cm/s
            vel_norm = np.linalg.norm(self.x.velocity(position_filtered[:, 0], position_filtered[:, 1],
                                                      unwrapped_angle_filtered * self.average_radius()), axis=1)

            stuck_frames = self.x.stuck(vel_norm=vel_norm)

            # plt.figure()
            # x_pos = position_filtered[:, 0]
            # plt.plot(x_pos)

            av_non_stuck_vel = np.mean(vel_norm[~np.array(stuck_frames).astype(bool)])
            d_stuck_per_frame = av_non_stuck_vel / self.x.fps
            stuck_path_length = np.sum(stuck_frames) * d_stuck_per_frame

            # plt.figure()
            # plt.plot(vel_norm)
            # plt.plot(stuck_frames*av_non_stuck_vel)
            # plt.show(block=False)
            DEBUG = 1

        else:
            stuck_path_length = 0

        if unwrapped_angle_filtered.size == 0 or position_filtered.size == 0:
            return 0

        aver_radius = self.average_radius()
        real_path_length = 0

        for pos1, pos2, ang1, ang2, stuck in \
                zip(position_filtered[:-1], position_filtered[1:],
                    unwrapped_angle_filtered[:-1], unwrapped_angle_filtered[1:],
                    stuck_frames):

            if not stuck:
                d = self.measureDistance(pos1, pos2, ang1, ang2, aver_radius, rot=rot)
                real_path_length += d

            if real_path_length + stuck_path_length > max_path_length:
                return real_path_length + stuck_path_length

        return real_path_length + stuck_path_length

    def plot(self, rot=True):
        plt.plot(self.x.position[:, 0], self.x.position[:, 1], color='blue')
        pos_list, ang_list = [], []

        unwrapped_angle = ConnectAngle(self.x.angle[1:], self.x.shape)
        pos, ang = self.x.position[0], unwrapped_angle[0]
        for i in range(len(unwrapped_angle)):
            d = self.measureDistance(pos, self.x.position[i], ang, unwrapped_angle[i], self.average_radius(), rot=rot)
            if d > resolution(self.x.geometry(), self.x.size, self.x.solver, self.x.shape):
                pos_list.append(pos)
                ang_list.append(ang)
        plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1], color='k')
        plt.show()

    def minimal(self) -> float:
        if self.x.shape in ['SPT']:
            ideal_filename = Path_planning_full_knowledge.minimal_filename(self.x, self.x.initial_cond())
            ideal = get(ideal_filename)
            return PathLength(ideal).per_experiment() * \
                   Maze(self.x).exit_size/Maze(ideal, geometry=self.x.geometry).exit_size
        else:
            return np.nan

    def comparable(self, maximal=25) -> tuple:
        """
        Cut experiment after certain path length distance which scales with group size.
        (10 times the minimal path length)
        Adjust winner boolean.
        """
        max_path_length = self.minimal() * maximal
        path_length = self.calculate_path_length(max_path_length=max_path_length, penalize=False)
        winner = (path_length < max_path_length) and self.x.winner
        return path_length, winner

    @classmethod
    def create_dicts(cls, myDataFrame):
        dictio_p = {}
        dictio_pp = {}
        # myDataFrame = myDataFrame[myDataFrame['solver'] == 'ant']
        # myDataFrame = myDataFrame[myDataFrame['size'] == 'S']
        for filename in tqdm(myDataFrame['filename']):
            print(filename)
            x = get(filename)
            dictio_p[filename] = PathLength(x).calculate_path_length(penalize=False)
            dictio_pp[filename] = PathLength(x).calculate_path_length(penalize=True)
        return dictio_p, dictio_pp


if __name__ == '__main__':
    # filename = 'L_LASH_4160019_LargeLH_1_ants (part 1)'
    # x = get(filename)
    # print(PathLength(x).calculate_path_length(penalize=True))
    # filename = 'S_SPT_4710014_SSpecialT_1_ants (part 1)'
    # filename = 'M_SPT_4700022_MSpecialT_1_ants'
    # filename = 'large_20201220135801_20201220140247'
    # filename = 'small_20220606175207_20220606175429'
    # filename = 'YI029806_5610'
    # x = get(filename)
    # print(PathLength(x).calculate_path_length(penalize=True))

    dictio_p, dictio_pp = PathLength.create_dicts(myDataFrame)
    DEBUG = 1
    with open(path_length_dir, 'w') as json_file:
        json.dump(dictio_p, json_file)
        json_file.close()

    with open(penalized_path_length_dir, 'w') as json_file:
        json.dump(dictio_pp, json_file)
        json_file.close()
    DEBUG = 1


with open(path_length_dir, 'r') as json_file:
    path_length_dict = json.load(json_file)
    json_file.close()

with open(penalized_path_length_dir, 'r') as json_file:
    penalized_path_length_dict = json.load(json_file)
    json_file.close()
