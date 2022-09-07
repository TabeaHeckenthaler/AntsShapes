from copy import copy
from trajectory_inheritance.get import get
from matplotlib import pyplot as plt
from Setup.Maze import Maze
from PS_Search_Algorithms.Path_planning_full_knowledge import Path_planning_full_knowledge
from Analysis.average_carrier_number.averageCarrierNumber import myDataFrame
from tqdm import tqdm
from Directories import first_frame_dir
import json
import numpy as np


class FirstFrame:
    def __init__(self, x):
        self.x = copy(x)

    def calculate_first_frame(self) -> int:
        """
        Reduce path to a list of points that each have distance of at least resolution = 0.1cm
        to the next point.
        Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
        Path length the sum of the distances of the points in the list.
        When the shape is standing still, the path length increases. Penalizing for being stuck.
        """
        # TODO: What is first frame for humans?
        v_norm = np.linalg.norm(self.x.velocity(), axis=1)
        # plt.plot(v_norm)
        # plt.show(block=False)

        v_first_frame = np.mean(v_norm) * 0.4

        # kernel_size = 2 * (self.x.fps // 2) + 1
        # position_filtered, unwrapped_angle_filtered = self.x.smoothed_pos_angle(self.x.position, self.x.angle,
        #                                                                         kernel_size)

        return int(np.where(v_norm > v_first_frame)[0][0])

    @classmethod
    def create_dicts(cls, myDataFrame) -> dict:
        first_frame_dict = {}
        for filename in tqdm(myDataFrame['filename']):
            print(filename)
            x = get(filename)
            first_frame_dict[filename] = cls(x).calculate_first_frame()
        return first_frame_dict

    @staticmethod
    def add_to_dict(myDataFrame, first_frame_dict: dict) -> dict:
        to_add = set(myDataFrame['filename']) - set(first_frame_dict.keys())
        for filename in to_add:
            x = get(filename)
            first_frame_dict.update({filename: FirstFrame(x).calculate_first_frame()})
        return first_frame_dict

    @staticmethod
    def get_dict() -> dict:
        with open(first_frame_dir, 'r') as json_file:
            first_frame_dict = json.load(json_file)
            json_file.close()
        return first_frame_dict

    @staticmethod
    def save_dict(first_frame_dict: dict) -> None:
        with open(first_frame_dir, 'w') as json_file:
            json.dump(first_frame_dict, json_file)
            json_file.close()


if __name__ == '__main__':
    # filename = 'L_SPT_5030006_LSpecialT_1_ants (part 1)'
    # x = get(filename)
    # print(FirstFrame(x).calculate_first_frame())

    # first_frame_dict = FirstFrame.get_dict()
    first_frame_dict = FirstFrame.create_dicts(myDataFrame)
    # first_frame_dict = FirstFrame.add_to_dict(myDataFrame, first_frame_dict)
    FirstFrame.save_dict(first_frame_dict)


