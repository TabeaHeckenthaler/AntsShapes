import numpy as np

from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from trajectory_inheritance.get import get
import json
import os
import pandas as pd
from Analysis.Efficiency.PathLength import PathLength
from Directories import network_dir
from tqdm import tqdm
from DataFrame.import_excel_dfs import find_minimal_pL, dfs_ant, dfs_ant_old, dfs_human, df_minimal
from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig

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
    def calc(cls, df, time_series_dict):
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

    @staticmethod
    def add_minimal(df, df_results):
        df['minimal'] = df.apply(lambda x: find_minimal_pL(x), axis=1)
        minimal_dict = {file: d for file, d in zip(df['filename'], df['minimal'])}
        df_results['minimal pL'] = df_results['filename'].map(minimal_dict)
        df_results['norm pL'] = df_results['pL'] / df_results['minimal pL']
        return df_results

    @staticmethod
    def add_grouptype(dfs, df_results):
        d = {}
        for grouptype, df in dfs.items():
            d.update({filename: grouptype for filename in df['filename']})
        df_results = df_results.assign(grouptype=df_results['filename'].map(d))
        return df_results

    @classmethod
    def plot(cls, results):
        y_err = results.groupby(['state', 'grouptype']).std().unstack()['norm pL']/len(results.groupby(['state', 'grouptype']).std().unstack())
        # sort before plotting by grouptype
        grouptypes = ['XL', 'L', 'M', 'S', 'Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']
        results['grouptype'] = pd.Categorical(results['grouptype'], grouptypes)
        ax = results.groupby(['state', 'grouptype']).mean().unstack().plot(kind='bar', y='norm pL', yerr=y_err, capsize=2)
        ax.set(xlabel='State', ylabel='Normalized Path Length')

if __name__ == '__main__':
    # create a pandas dataframe with all the slowdowns

    df = pd.concat(dfs_ant)

    # with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    #     time_series_dict = json.load(json_file)
    #     json_file.close()
    # results = Traj_sep_by_state_pL.calc(df, time_series_dict)
    # results.to_excel('pathL_per_state_ant.xlsx')

    results_ants = pd.read_excel('pathL_per_state_ant.xlsx')
    results_ants = Traj_sep_by_state_pL.add_minimal(df, results_ants)
    results_ants['grouptype'] = results_ants['size']
    # print(np.unique(results['grouptype']))
    # Traj_sep_by_state_pL.plot(results)
    # save_fig(plt.gcf(), 'pathL_per_state_ant.png')

    # __________________________________________________________________________________________________________________

    df = pd.concat(dfs_human)

    # with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    #     time_series_dict = json.load(json_file)
    #     json_file.close()
    # results = Traj_sep_by_state_pL.calc(df, time_series_dict)
    # results.to_excel('pathL_per_state_human.xlsx')

    results_humans = pd.read_excel('pathL_per_state_human.xlsx')
    results_humans = Traj_sep_by_state_pL.add_minimal(df, results_humans)
    results_humans = Traj_sep_by_state_pL.add_grouptype(dfs_human, results_humans)
    # Traj_sep_by_state_pL.plot(results)
    # save_fig(plt.gcf(), 'pathL_per_state_human.png')

    # __________________________________________________________________________________________________________________
    results = pd.concat([results_ants, results_humans])
    Traj_sep_by_state_pL.plot(results)
    save_fig(plt.gcf(), 'pathL_per_state.png')