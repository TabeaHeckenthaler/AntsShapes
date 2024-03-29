from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from trajectory_inheritance.get import get
from tqdm import tqdm
import pandas as pd
from Directories import network_dir, home
from mayavi import mlab
import os
from DataFrame.import_excel_dfs import dfs_human, dfs_ant
import json
import numpy as np
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}
columns = ['filename', 'size', 'solver', 'state', 'extremal point', 'frame']


class ExtremalPoints:
    def __init__(self, df: pd.DataFrame, unique_states: list, succession: list):
        self.df = df
        self.unique_states = unique_states
        self.succession = succession

    @staticmethod
    def find_minimal_point_in_x(traj_part) -> tuple:
        minimal_i = np.argmin(traj_part.position[:, 0])
        minimal_coordinates = traj_part.position[minimal_i].tolist() + [traj_part.angle[minimal_i]]
        frame = traj_part.parent_traj.frames[traj_part.frames_of_parent[0] + minimal_i]
        return minimal_coordinates, frame

    def get_coordinates(self):
        coords = self.df['extremal point'].map(ExtremalPoints.to_list).tolist()
        return coords

    def plot_in_cs(self, cs: ConfigSpace_Maze, coords: list):
        scale_factor = {'XL': 0.2, 'L': 0.1, 'M': 0.05, 'S': 0.03, 'Large': 1., 'Medium': 0.5, 'Small Far': 0.2,
                        'Small Near': 0.2, 'Small': 0.2}
        for coord in coords:
            cs.draw(coord[:2], coord[2], scale_factor=scale_factor[cs.size])

    @staticmethod
    def to_list(string: str):
        return [eval(x) for x in string.strip('][').split(', ') if len(x) > 0]

    def calc_extremal_points(self) -> pd.DataFrame:
        new_results = pd.DataFrame(columns=columns)

        for filename in tqdm(self.df['filename']):
            x = get(filename)
            print(x.filename)

            ts = time_series_dict[x.filename]
            ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)
            successions = Traj_sep_by_state(x, ts_extended).get_successions_of_states(self.succession)

            for s in successions:
                x_suc = s[0][0]
                for t in s[0]:
                    x_suc = x_suc + t

            traj_parts = Traj_sep_by_state(x, ts_extended).get_states(wanted_states=self.unique_states)

            for traj_part in traj_parts:
                extremal_point, frame = self.find_minimal_point_in_x(traj_part)
                d = {'filename': x.filename, 'size': x.size, 'solver': x.solver, 'state': list(set(traj_part.states)),
                     'extremal point': extremal_point, 'frame': frame}
                new_results = new_results.append(d, ignore_index=True)
        return new_results


if __name__ == '__main__':

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    unique_s = ['ab', 'ac']
    necessary_succ = [['ab'], ['b', 'b1', 'b2', 'be'], ['ab'], ['ac'], ['c']]
    e = ExtremalPoints(df=pd.concat(dfs_human), unique_states=unique_s, succession=necessary_succ)
    new_results = e.calc_extremal_points()
    new_results.to_excel(os.path.join(home, 'Analysis', 'discretisation', 'extremal_points_ab_ac_human' + '.xlsx'))

    directory = os.path.join(home, 'Analysis', 'discretisation', 'extremal_points_ab_ac_human' + '.xlsx')
    e_p_results = pd.read_excel(directory, usecols=columns)

    for size, df_size in dfs_human.items():
        df = e_p_results[e_p_results['filename'].isin(df_size['filename'])]
        extr_points = ExtremalPoints(unique_states=unique_s, df=df)
        cs = ConfigSpace_Maze('human', df.iloc[0]['size'], 'SPT',
                              ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
        cs.visualize_space()
        coords = extr_points.get_coordinates()
        # coords = e_p_results[e_p_results['size'] == size]['extremal point'].map(ExtremalPoints.to_list).tolist()
        extr_points.plot_in_cs(cs, coords)

        mlab.show()
        DEBUG = 1

    # # how many percentage of times the shape entered ... it did this succession?
    # columns = ['filename', 'size', 'solver', 'start', 'end', 'percent']
    # new_results = pd.DataFrame(columns=columns)
    #
    # for size, df in dfs_human.items():
    #     for filename in tqdm(df['filename']):
    #         x = get(filename)
    #         print(x.filename)
    #
    #         ts = time_series_dict[x.filename]
    #         ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)
    #         t_sep = Traj_sep_by_state(x, ts_extended)
    #         start, end = necessary_succ[:2], necessary_succ[2:]
    #         perc = t_sep.percent_of_succession1_ended_like_succession2(succession1=start,
    #                                                                    succession2=start+end)
    #         d = {'filename': x.filename, 'size': x.size, 'solver': x.solver,
    #              'start': start, 'end': end, 'percent': perc}
    #         new_results = new_results.append(d, ignore_index=True)
    #
    # new_results.to_excel(os.path.join(home, 'Analysis', 'discretisation', 'perc_of_succession_ending_' + unique_s + '.xlsx'))


