from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from trajectory_inheritance.get import get
from tqdm import tqdm
import pandas as pd
from Directories import network_dir, home
import os
from DataFrame.import_excel_dfs import dfs_human, dfs_ant
import json
from Analysis.Efficiency.PathLength import PathLength
import matplotlib.pyplot as plt
import numpy as np

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}
columns = ['filename', 'size', 'solver', 'state', 'turning radius (norm by 2 pi)']


class BackRoomTurning:
    def __init__(self, unique_states: list):
        self.unique_states = unique_states

    def plot(self, results, title: str):
        # plot results in columns 'turning radius (norm by 2 pi)' in a histogram
        for size, df_size in results.groupby('size'):
            print(size)
            fig, axs = plt.subplots(2)

            if solver == 'human':
                limits = [[0, 1], [0, 1]]
            elif solver == 'ant':
                limits = [[0, 1], [0, 16]]

            for (states, df_state_size), ax, lim in zip(df_size.groupby('state'), axs, limits):
                theta = df_state_size['turning radius (norm by 2 pi)']
                ax.hist(theta, bins=20)
                ax.set_xlabel('turning radius (norm by 2 pi)')
                ax.set_ylabel('count')
                ax.set_title(size + ', passed through states: ' + str(states))
                ax.set_xlim(*lim)

            # plt.show()
            plt.tight_layout()
            plt.savefig('images\\turning_radius\\' + title + size + '.png')
        DEBUG = 1

    @staticmethod
    def to_list(string: str):
        return [eval(x) for x in string.strip('][').split(', ') if len(x) > 0]

    def calc_turning_angle(self, df) -> pd.DataFrame:
        new_results = pd.DataFrame(columns=columns)
        for filename in tqdm(df['filename']):
            x = get(filename)
            print(x.filename)

            ts = time_series_dict[x.filename]
            ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)
            traj_parts = Traj_sep_by_state(x, ts_extended).get_states(wanted_states=self.unique_states)

            for traj_part in traj_parts:
                d_theta = PathLength(traj_part).rotational_distance(norm=False)
                print(filename, round(d_theta / (2 * np.pi), 3), list(set(traj_part.states)))
                d = {'filename': filename, 'size': x.size, 'solver': x.solver,
                     'state': list(set(traj_part.states)),
                     'turning radius (norm by 2 pi)': round(d_theta / (2 * np.pi), 3),
                     }
                new_results = new_results.append(d, ignore_index=True)
        return new_results

    def calc_turning_angle_intermediate(self, df) -> pd.DataFrame:
        new_results = pd.DataFrame(columns=columns)
        for filename in tqdm(df['filename']):
            # if filename == 'M_SPT_5050016_MSpecialT_1':
                x = get(filename)
                print(x.filename)

                ts = time_series_dict[x.filename]
                ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)
                x.smooth(sec_smooth=1)
                traj_parts = Traj_sep_by_state(x, ts_extended).get_states(wanted_states=self.unique_states)
                traj_parts = [t.split_at_directional_change_in_turning() for t in traj_parts]
                traj_parts = [item for sublist in traj_parts for item in sublist]

                for traj_part in tqdm(traj_parts, desc=filename):
                    d_theta = PathLength(traj_part).rotational_distance(norm=False, smooth=False)
                    print(filename, round(d_theta / (2 * np.pi), 3), list(set(traj_part.states)))
                    d = {'filename': filename, 'size': x.size, 'solver': x.solver,
                         'state': list(set(traj_part.states)),
                         'turning radius (norm by 2 pi)': round(d_theta / (2 * np.pi), 3),
                         }
                    new_results = new_results.append(d, ignore_index=True)
        return new_results


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()
    unique_s = ['ab', 'ac']
    brt = BackRoomTurning(unique_states=unique_s)

    for solver, dfs in [['ant', dfs_ant], ['human', dfs_human], ]:
        # turning radius when in unique states
        # directory = os.path.join(home, 'Analysis', 'discretisation', 'overturning_in_ab_ac_humans.xlsx')
        # new_results = brt.calc_turning_angle(pd.concat(dfs_ant))
        # new_results.to_excel(directory)

        # turning radius when in unique states and turning in the same direction
        directory = os.path.join(home, 'Analysis', 'discretisation',
                                 'overturning_in_ab_ac_' + solver + '_intermediate.xlsx')
        # new_results = brt.calc_turning_angle_intermediate(pd.concat(dfs))
        # new_results.to_excel(directory)

        results = pd.read_excel(directory, usecols=columns)

        results['size'] = results['size'].replace('Small Far', 'Small')
        results['size'] = results['size'].replace('Small Near', 'Small')

        brt.plot(results, 'turning_radius_' + solver + '_')
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
