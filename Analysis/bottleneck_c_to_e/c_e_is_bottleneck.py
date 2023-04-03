import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import pyplot as plt
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.Efficiency.PathLength import PathLength
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
import os
import json
from tqdm import tqdm
from Directories import network_dir, home, minimal_path_length_dir
from trajectory_inheritance.trajectory import Trajectory_part
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from DataFrame.import_excel_dfs import dfs_ant, df_minimal, dfs_ant_old, dfs_human, df_human
from DataFrame.gillespie_dataFrame import dfs_gillespie
from plotly import express as px
from ConfigSpace.experiment_sliding import Experiment_Sliding
from Analysis.greed import Trajectory
import pandas as pd
from correlation_edge_walk_decision_c_e_ac import In_the_bottle


def cut_traj(traj, ts, buffer=0) -> tuple:
    """
    Divide the trajectory into two lists.
    First, cut off the exp after the last c state.
    Second, split traj into subtrajs that are (1) outside c and cg and (2) within c and cg.
    Return these two lists.
    """
    ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, traj)
    indices_cg = np.where(np.logical_or(np.array(ts_extended) == 'c', np.array(ts_extended) == 'cg'))[0]
    ind_succ = np.split(indices_cg, np.where(np.diff(indices_cg) != 1)[0] + 1)

    in_cg = np.zeros_like(traj.frames).astype(bool)
    in_cg[np.hstack(ind_succ)] = True

    where_in_cg = np.array([i for i, x in enumerate(in_cg) if x])
    index_successions_in_cg = np.split(where_in_cg, np.where(np.diff(where_in_cg) != 1)[0] + 1)

    where_out_cg = np.array([i for i, x in enumerate(in_cg) if not x])
    index_successions_out_cg = np.split(where_out_cg, np.where(np.diff(where_out_cg) != 1)[0] + 1)

    in_c_trajs = [Trajectory_part(traj, indices=[i[0] - buffer * traj.fps, i[-1] + buffer * traj.fps], VideoChain=[],
                                  tracked_frames=[], parent_states=ts_extended)
                  for i in index_successions_in_cg if len(i) > 0]
    out_c_trajs = [Trajectory_part(traj, indices=[i[0] - buffer * traj.fps, i[-1] + buffer * traj.fps], VideoChain=[],
                                   tracked_frames=[], parent_states=ts_extended)
                   for i in index_successions_out_cg if len(i) > 0]
    return in_c_trajs, out_c_trajs


def pL(traj):
    return PathLength(traj).translational_distance(smooth=False) + PathLength(traj).rotational_distance(smooth=False)


def cut_traj_after_c_e_crossing(traj) -> Trajectory_part:
    """
    Divide the trajectory into two lists.
    First, cut off the exp after the last c state.
    Second, split traj into subtrajs that are (1) outside c and cg and (2) within c and cg.
    Return these two lists.
    """
    ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(time_series_dict[traj.filename], traj)
    indices_e = np.where(np.array(ts_extended) == 'e')[0]
    return Trajectory_part(traj, indices=[indices_e[0], -1], VideoChain=[], tracked_frames=[])


def calc(traj):
    if 'e' in time_series_dict[traj.filename]:
        after_e = cut_traj_after_c_e_crossing(traj)
        return pL(after_e)
    else:
        return None


def plot_means():
    median_pL_after_first_e, std_pL_after_first_e = {}, {}

    for size, df in df_dict_sep_by_size.items():
        print(size)
        minimal = df_minimal[(df_minimal['size'] == df.iloc[0]['size']) & (df_minimal['shape'] == 'SPT') & (
                df_minimal['initial condition'] == 'back')].iloc[0]['path length [length unit]']

        df['pL_after_first_e'] = df['filename'].map(pL_after_first_e)

        pL_after_first_e_size = [item / minimal for item in df['pL_after_first_e'].dropna()]
        median_pL_after_first_e[size] = np.mean(pL_after_first_e_size)
        std_pL_after_first_e[size] = np.std(pL_after_first_e_size)

    fig_pL_after_first_e = px.bar(x=list(median_pL_after_first_e.keys()),
                                  y=list(median_pL_after_first_e.values()),
                                  error_y=list(std_pL_after_first_e.values()))
    fig_pL_after_first_e.update_yaxes(title_text='mean_pL_after_first_e/minimal pL')
    fig_pL_after_first_e.update_xaxes(title_text='size')

    save_fig(fig_pL_after_first_e, 'pL_after_first_e' + _add)


def plot_percentage():
    median_pL_after_first_e, std_pL_after_first_e = {}, {}

    for size, df in df_dict_sep_by_size.items():
        print(size)
        df = df[df['winner']]

        df['rotation'] = df['filename'].map(rotation_dict)
        df['translation'] = df['filename'].map(translation_dict)
        df['total_pL'] = df['rotation'] + df['translation']
        df['pL_after_first_e'] = df['filename'].map(pL_after_first_e)
        df['percentage'] = df['pL_after_first_e'] / df['total_pL'].tolist()
        percentage = df['percentage'].dropna().tolist()

        median_pL_after_first_e[size] = np.median(percentage)
        std_pL_after_first_e[size] = np.std(percentage)

    plt.errorbar(median_pL_after_first_e.keys(),
                 median_pL_after_first_e.values(),
                 yerr=std_pL_after_first_e.values())
    plt.xlabel('size')
    plt.ylabel('pL_after_first_e/total_pL')
    save_fig(plt.gcf(), 'pL_after_first_e_percentage' + _add)

def wo_klemmts(filename):
    ts = time_series_dict[filename]


if __name__ == '__main__':
    filename = 'large_20210805171741_20210805172610'

    with open(minimal_path_length_dir, 'r') as json_file:
        minimal_path_length_dict = json.load(json_file)
        json_file.close()

    df_human['minimal'] = df_human['filename'].map(minimal_path_length_dict)
    wo_klemmts(filename)

    solver, shape = 'ant', 'SPT'
    df_dict_sep_by_size = dfs_ant

    with open(os.path.join(network_dir, 'time_series_selected_states' + '.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    radius = 10
    _add = '_' + solver + str(radius)

    # pL_after_first_e = {}
    # for filename in tqdm(pd.concat(dfs_human)['filename']):
    #     traj = get(filename)
    #     pL_after_first_e[filename] = calc(traj)

    # with open('results\\pL_after_first_e' + _add + '.json', 'w') as f:
    #     json.dump(pL_after_first_e, f)

    with open('results\\pL_after_first_e' + _add + '.json', 'r') as f:
        pL_after_first_e = json.load(f)

    translation_dir = os.path.join(home, 'Analysis', 'Efficiency', 'translation.json')
    rotation_dir = os.path.join(home, 'Analysis', 'Efficiency', 'rotation.json')
    minimal_filename_dict = os.path.join(home, 'Analysis', 'minimal_path_length', 'minimal_filename_dict.json')

    with open(translation_dir, 'r') as json_file:
        translation_dict = json.load(json_file)
        json_file.close()

    with open(rotation_dir, 'r') as json_file:
        rotation_dict = json.load(json_file)
        json_file.close()

    # plot_means()
    plot_percentage()
