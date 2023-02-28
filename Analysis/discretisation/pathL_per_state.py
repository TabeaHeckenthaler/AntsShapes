from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from trajectory_inheritance.get import get
import json
import os
import pandas as pd
from Analysis.Efficiency.PathLength import PathLength
from Directories import network_dir
from tqdm import tqdm
from DataFrame.import_excel_dfs import find_minimal, dfs_ant, dfs_ant_old, dfs_human

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}

columns = ['filename', 'size', 'solver', 'state', 'frames', 'pL', 'time']


class Traj_sep_by_state_pL(Traj_sep_by_state):
    def to_df(self):
        df = pd.DataFrame(columns=columns)
        for traj_part, state in zip(self.traj_parts, self.states):
            d = {'filename': traj_part.filename, 'size': traj_part.size, 'solver': traj_part.solver,
                 'state': state, 'frames': [traj_part.frames[0], traj_part.frames[-1]],
                 'pL': PathLength(traj_part).total_dist(), 'time': traj_part.timer()}
            df = df.append(d, ignore_index=True)
        return df

    @classmethod
    def calc(cls, df):
        new_results = pd.DataFrame(columns=columns)
        for filename in tqdm(df['filename']):
            # filename = 'large_20210805171741_20210805172610'
            print(filename)
            x = get(filename)
            ts = time_series_dict[x.filename]
            ts_extended = cls.extend_time_series_to_match_frames(ts, x)
            traj = cls(x, ts_extended)

            traj.find_state_change_indices()
            d = traj.to_df()
            new_results = new_results.append(d, ignore_index=True)
            # print(d)
            DEBUG = 1
        return new_results


if __name__ == '__main__':
    # create a pandas dataframe with all the slowdowns

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    df = pd.concat(dfs_ant)
    results = Traj_sep_by_state_pL.calc(df)
    # save new results in excel file
    results.to_excel('pathL_per_state_ant.xlsx')
    #
    # with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    #     time_series_dict = json.load(json_file)
    #     json_file.close()
    df = pd.concat(dfs_human)
    results = Traj_sep_by_state_pL.calc(df)
    results.to_excel('pathL_per_state_human.xlsx')

