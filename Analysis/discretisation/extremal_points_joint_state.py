from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from trajectory_inheritance.get import get
from tqdm import tqdm
import pandas as pd
from Directories import network_dir, home
from mayavi import mlab
import os
from DataFrame.import_excel_dfs import dfs_human
import json
import numpy as np
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}
columns = ['filename', 'size', 'solver', 'state', 'extremal point', 'frame']


class ExtremalPoints:
    def __init__(self, df: pd.DataFrame, unique_state: str, succession: list):
        self.df = df
        self.unique_state = unique_state
        self.succession = succession

    @staticmethod
    def find_minimal_point_in_x(traj_part) -> tuple:
        minimal_i = np.argmin(traj_part.position[:, 0])
        minimal_coordinates = traj_part.position[minimal_i].tolist() + [traj_part.angle[minimal_i]]
        frame = traj_part.parent_traj.frames[traj_part.frames_of_parent[0] + minimal_i]
        return minimal_coordinates, frame

    def get_coordinates(self):
        coords = self.df[self.df['size'] == size]['extremal point'].map(ExtremalPoints.to_list).tolist()
        return coords

    def plot_in_cs(self, cs: ConfigSpace_Maze):
        scale_factor = {'XL': 0.2, 'L': 0.1, 'M': 0.05, 'S': 0.03, 'Large': 1., 'Medium': 0.5, 'Small Far': 0.2,
                        'Small Near': 0.2, 'Small': 0.2}
        for coord in self.get_coordinates():
            cs.draw(coord[:2], coord[2], scale_factor=scale_factor[cs.size])

    @staticmethod
    def to_list(string: str):
        return [eval(x) for x in string.strip('][').split(', ') if len(x) > 0]

    def calc_extremal_points(self, df, unique_state) -> pd.DataFrame:
        new_results = pd.DataFrame(columns=columns)
        for filename in tqdm(df['filename']):
            x = get(filename)
            print(x.filename)

            ts = time_series_dict[x.filename]
            ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)
            successions = Traj_sep_by_state(x, ts_extended).get_successions_of_states(self.succession)

            for (traj_parts, states) in successions:
                for i in [i for i, state in enumerate(states) if state == self.unique_state]:
                    extremal_point, frame = self.find_minimal_point_in_x(traj_parts[i])
                    d = {'filename': x.filename, 'size': x.size, 'solver': x.solver,
                         'state': unique_state, 'extremal point': extremal_point,
                         'frame': frame}
                    new_results = new_results.append(d, ignore_index=True)
        return new_results

    def percentage_of_entrances_to_unique_state_followed_succession(self):

        pass


if __name__ == '__main__':

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    unique_s = 'ab'
    necessary_succ = [['ab'], ['b', 'b1', 'b2', 'be'], ['ab'], ['ac'], ['c']]
    e = ExtremalPoints(coordinates=[], unique_state=unique_s, succession=necessary_succ)
    # new_results = e.calc_extremal_points(pd.concat(dfs_human))
    # new_results.to_excel(os.path.join(home, 'Analysis', 'discretisation', 'extremal_points_' + unique_s + '.xlsx'))

    directory = os.path.join(home, 'Analysis', 'discretisation', 'extremal_points_' + unique_s + '.xlsx')
    results = pd.read_excel(directory, usecols=columns)

    # for size, df in dfs_human.items():
    #     coords = e_p[e_p['size'] == size]['extremal point'].map(ExtremalPoints.to_list).tolist()
    #
    #     extr_points = ExtremalPoints(coordinates=coords, unique_state=unique_s)
    #     cs = ConfigSpace_Maze('human', size, 'SPT',
    #                           ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
    #     cs.visualize_space()
    #     extr_points.plot_in_cs(cs)
    #
    #     mlab.show()
    #     DEBUG = 1

    # how many percentage of times the shape entered ... it did this succession?
    for size, df in dfs_human.items():
        e = ExtremalPoints(unique_state=unique_s, succession=necessary_succ, )
