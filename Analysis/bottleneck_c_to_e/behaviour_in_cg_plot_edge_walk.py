import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import seaborn as sns
import numpy as np
from mayavi import mlab
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.get import get
import os
import json
from Directories import network_dir
from trajectory_inheritance.trajectory import Trajectory_part
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from PIL import Image
from DataFrame.import_excel_dfs import dfs_ant, df_minimal, dfs_ant_old
from Analysis.Efficiency.PathLength import PathLength
from ConfigSpace.experiment_sliding import Experiment_Sliding


def extend_time_series_to_match_frames(ts):
    indices_to_ts_to_frames = np.cumsum([1 / (int(len(traj.frames) / len(ts) * 10) / 10)
                                         for _ in range(len(traj.frames))]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def cut_traj_to_cg_part(filename):
    traj = get(filename)
    ts_extended = extend_time_series_to_match_frames(time_series_dict[filename])
    indices_cg = np.where(np.logical_or(np.array(ts_extended) == 'c', np.array(ts_extended) == 'cg'))[0]
    index_successions = np.split(indices_cg, np.where(np.diff(indices_cg) != 1)[0] + 1)
    frames = [index_successions[0][0], index_successions[0][-1]]
    print(filename + str(frames[0]))

    traj_cg = Trajectory_part(traj, VideoChain=[], indices=frames, tracked_frames=[])
    return traj_cg


def plot_traj_in_cs(ps: ConfigSpace_Maze, traj: Trajectory_part, radius: int, bool_to_plot: list = None,
                    space_to_show=None):
    """
    Plot the trajectories in the config space.
    """
    num_to_plot = 1  # len(trajs)
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    colors = cmap(np.linspace(0.2, 1, num_to_plot))[:, :3]
    ps.visualize_space(space=space_to_show)

    if bool_to_plot is None:
        bool_to_plot = np.ones_like(traj.angle).astype(bool)
    pos = traj.position[bool_to_plot, :]
    ang = traj.angle[bool_to_plot]

    edge_walk = Experiment_Sliding(traj)
    edge_walk.find_on_edge(ps, radius)
    edge_walk.clean_exp_from_short_values()
    ind_succ = Experiment_Sliding.index_successions(edge_walk.on_edge)

    edge_boolean = np.zeros_like(traj.frames).astype(bool)
    edge_boolean[np.hstack(ind_succ)] = True

    where_on_edge = np.array([i for i, x in enumerate(edge_boolean) if x])
    index_successions_on_edge = np.split(where_on_edge, np.where(np.diff(where_on_edge) != 1)[0] + 1)

    where_off_edge = np.array([i for i, x in enumerate(edge_boolean) if not x])
    index_successions_off_edge = np.split(where_off_edge, np.where(np.diff(where_off_edge) != 1)[0] + 1)

    on_edge = [(i[0], i[-1]) for i in index_successions_on_edge if len(i) > 0]
    off_edge = [(i[0], i[-1]) for i in index_successions_off_edge if len(i) > 0]

    scale_factor = {'XL': 0.1, 'L': 0.05, 'M': 0.025, 'S': 0.0125, }

    for (start, end) in off_edge:
        ps.draw(pos[start:end, :], ang[start:end], scale_factor=scale_factor[ps.size], color=(0, 0, 0))
    for (start, end) in on_edge:
        ps.draw(pos[start:end, :], ang[start:end], scale_factor=scale_factor[ps.size], color=(1, 0, 0))

    mlab.show()


# arr = mlab.screenshot(ps.fig, mode='rgb')
# arr = mlab.screenshot(ps.fig, mode='rgb')
# im = Image.fromarray(arr)
# im.save('images\\' + traj.filename + str(frames[0]) + '.jpeg')
# mlab.close()

def pL(traj):
    return PathLength(traj).translational_distance() + PathLength(traj).rotational_distance()


# arr = mlab.screenshot(ps.fig, mode='rgb')
# arr = mlab.screenshot(ps.fig, mode='rgb')
# im = Image.fromarray(arr)
# im.save('images\\' + traj.filename + str(frames[0]) + '.jpeg')
# mlab.close()

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

if __name__ == '__main__':
    shape, solver, size, radius = 'SPT', 'ant', 'L', 10
    ps = ConfigSpace_SelectedStates(solver=solver, size=size, shape=shape, geometry=(
        'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
    ps.load_final_labeled_space()
    for filename in dfs_ant['L']['filename']:
        traj = get(filename)
        traj_cg = cut_traj_to_cg_part(filename)
        c_space = np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg')
        plot_traj_in_cs(ps=ps, traj=traj_cg, space_to_show=c_space, radius=radius)

        DEBUG = 1