import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from trajectory_inheritance.get import get
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates, ConfigSpace_Maze
from DataFrame.import_excel_dfs import dfs_ant
import seaborn as sns
import numpy as np
from mayavi import mlab
import json
import os
from plotly import express as px
from Analysis.Efficiency.PathLength import PathLength
import plotly.graph_objects as go
from copy import deepcopy
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from Analysis.PathPy.Path import Path, exp_day, color_dict
from Directories import network_dir, home
from DataFrame.plot_dataframe import save_fig
from matplotlib import pyplot as plt
from tqdm import tqdm
from trajectory_inheritance.trajectory import Trajectory_part
from typing import Union


results_dir = os.path.join('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\results\\')


class Sliding_Window:
    def __init__(self, traj: Trajectory_part, ts=None, frames=0):
        self.traj = traj
        self.time_series = ts
        self.frames = self.traj.frames

    def pathLength(self) -> float:
        return PathLength(self.traj).translational_distance() + PathLength(self.traj).rotational_distance()

    def timer(self):
        return self.traj.timer()

    def prominent_state(self):
        return max(set(self.time_series), key=self.time_series.count)


class Experiment_Sliding:
    def __init__(self, traj, boundary=None, on_edge=None, sliding=None, moving=None):
        self.boundary = boundary
        self.traj = traj
        self.on_edge = on_edge
        self.sliding = sliding
        self.moving = moving

    def new_traj(self, traj):
        self.traj = traj
        self.on_edge = None
        self.sliding = None
        self.moving = None

    def find_boundary(self, ps, radius) -> None:
        """
        Find the mask of the boundary of the maze.
        """
        boundary_and_forbidden = ps.dilate(space=~ps.space, radius=radius)
        # self.boundary = np.logical_and(boundary_and_forbidden, ps.space)
        self.boundary = boundary_and_forbidden

        # ps.visualize_space(space=self.boundary)
        # boundary.sum()/boundary.size
        # for traj, color in zip(trajs[:1], colors):
        #     ps.draw(traj.position, traj.angle, scale_factor=0.2, color=tuple(color))

    def distances_walked(self, boolean_list) -> tuple:
        """
        Find the distance walked on the edge.
        """
        trans = []
        rot = []

        index_successions = self.index_successions(boolean_list)

        for i_s in index_successions:
            if len(i_s) > self.traj.fps:
                trans.append(PathLength(self.traj).translational_distance(frames=[i_s[0], i_s[-1]]))
                rot.append(PathLength(self.traj).rotational_distance(frames=[i_s[0], i_s[-1]]))
        return rot, trans

    def fraction_sliding(self) -> float:
        vel = np.linalg.norm(self.traj.velocity(), axis=1)
        vel = np.hstack([vel, vel[-1]])

        sliding = np.sum(vel * np.array(self.sliding).astype(int)) * 1 / self.traj.fps
        all_motion = np.sum(vel) * 1 / self.traj.fps
        print(self.traj.filename, sliding / all_motion)
        return sliding / all_motion

    def find_on_edge(self, ps, radius=None) -> None:
        """
        Boolean array whether traj is on the boundary
        """
        on_edge = np.zeros_like(self.traj.angle).astype(bool)
        if self.boundary is None:
            self.find_boundary(ps, radius)

        # ps.visualize_space(space=self.boundary)
        # ps.draw(self.traj.position[on_edge], self.traj.angle[on_edge], scale_factor=0.05)
        # ps.draw_ind(self.traj.position, self.traj.angle, scale_factor=0.05)

        for i in range(self.traj.position.shape[0]):
            indices = ps.coords_to_indices(x=self.traj.position[i, 0], y=self.traj.position[i, 1],
                                           theta=self.traj.angle[i])
            # ps.visualize_space(space=self.boundary)
            # ps.draw_ind(indices, scale_factor=0.05)
            if self.boundary[indices]:
                on_edge[i] = True
        self.on_edge = on_edge

    def play_edge_walk(self) -> None:
        x = deepcopy(self.traj)
        my_maze = Maze(x)
        display = Display(x.filename, x.fps, my_maze, wait=10)

        i = 0

        while i < len(x.frames):
            if self.on_edge[i]:
                color_background = (255, 224, 224)
            else:
                color_background = (250, 250, 250)
            display.renew_screen(movie_name=x.filename,
                                 frame_index=str(x.frames[display.i]),
                                 color_background=color_background)
            x.step(my_maze, i, display=display)

            if display is not None:
                end = display.update_screen(x, i)
                if end:
                    display.end_screen()
                    x.frames = x.frames[:i]
                    break
            i += 1
        if display is not None:
            display.end_screen()

    def plot(self, traj) -> None:
        index_succession = self.index_successions(self.on_edge)
        if self.traj.timer() * len(np.concatenate(index_succession)) / len(traj.frames) / len(index_succession) > 20:
            vel = np.linalg.norm(self.traj.velocity(), axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.traj.frames, y=np.array(self.on_edge).astype(int) * np.mean(vel)))
            fig.add_trace(go.Scatter(x=self.traj.frames, y=vel))
            fig.show()

    @staticmethod
    def fraction_time(boolean_list) -> float:
        return np.sum(np.array(boolean_list).astype(int)) / len(boolean_list)

    @staticmethod
    def index_successions(boolean_list) -> np.array:
        """
        Find the consecutive indices of the edge wall
        """
        where_on_edge = np.where(boolean_list)[0]
        index_successions = np.split(where_on_edge, np.where(np.diff(where_on_edge) != 1)[0] + 1)
        return index_successions

    @staticmethod
    def plot_means(dictionary) -> go.Figure:
        means, std = {}, {}

        sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL']
        for size in sizes:
            df_size = dfs_ant[size]
            df_size['to_plot'] = df_size['filename'].map(dictionary)

            means[size] = df_size['to_plot'].mean()
            std[size] = df_size['to_plot'].std()
        fig = px.bar(x=list(means.keys()), y=list(means.values()), error_y=list(std.values()))
        return fig

    @staticmethod
    def plot_distribution(dictionary):
        fig = go.Figure()
        for size, df_size in dfs_ant.items():
            df_size['to_plot'] = df_size['filename'].map(dictionary)
            y_data = np.concatenate(df_size['to_plot'].tolist())
            hist, bins = np.histogram(y_data, bins=20, density=True, range=(0, 10))
            fig.add_trace(go.Scatter(x=bins, y=hist, name=size))
        return fig

    @staticmethod
    def plot_bar_chart(df: pd.DataFrame, block=False) -> plt.Figure:
        fig, ax = plt.subplots()

        with open(results_dir + 'edge_walk.json', 'r') as f:
            edge_walk_dict = json.load(f)

        with open(results_dir + 'moving_dict.json', 'r') as f:
            moving_dict = json.load(f)

        df['time series'] = df['filename'].map(time_series_dict)
        df['edge walk'] = df['filename'].map(edge_walk_dict)
        df['moving'] = df['filename'].map(moving_dict)

        # for filename, ts, winner, food in zip(df['filename'], df['time series'], df['winner'], df['food in back']):
        for filename, ts, ew, m in zip(df['filename'], df['time series'], df['edge walk'], df['moving']):
            p = Path(time_step=0.25, time_series=ts)
            print(filename)

            mask = np.diff(np.cumsum([1 / (int(len(ew) / len(ts) * 10) / 10) for _ in range(len(ew))]).astype(int),
                           prepend=0)

            ew_shortened = ['edge-' if b else 'free-' for b in np.array(ew)[mask.astype(bool)]]
            m_shortened = ['moving' if b else 'still' for b in np.array(m + [m[-1]])[mask.astype(bool)]]

            # four colors
            sliding = [''.join(z) for z in zip(ew_shortened, m_shortened)]

            # p.bar_chart(ax=ax, axis_label=exp_day(filename), block=block)
            # p.bar_chart(ax=ax, axis_label=exp_day(filename) + '_in_contact', array=ew_shortened)
            # p.bar_chart(ax=ax, axis_label=exp_day(filename) + '_moving', array=m_shortened)
            p.bar_chart(ax=ax, axis_label=exp_day(filename) + '_moving', array=sliding)

            if not block:
                ax.set_xlabel('time [min]')
            else:
                ax.set_xlabel('')
            # ax.set_xlim([0, 20])

        plt.subplots_adjust(hspace=.0)
        return plt.gcf()

    @staticmethod
    def plot_traj_in_cs(ps: ConfigSpace_SelectedStates, traj, bool_to_plot: Union[list, None]):
        """
        Plot the trajectories in the config space.
        """
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        # colors = cmap(np.linspace(0.2, 1, len(trajs) * 2))[:, :3]
        scale_factor = {'XL': 0.2, 'L': 0.1, 'M': 0.05, 'S': 0.03}
        ps.visualize_space(space=np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg'), reduction=2)
        if bool_to_plot is None:
            bool_to_plot = np.ones_like(traj.angle).astype(bool)

        if traj.solver == 'gillespie':
            position = traj.position * {'XL': 1 / 4, 'L': 1 / 2, 'M': 1, 'S': 2}[traj.size]
        else:
            position = traj.position

        for bool_to_plot_, color in zip([bool_to_plot, np.logical_not(bool_to_plot)],
                                        [(0.96298491, 0.6126247, 0.45145074),
                                         (0.01060815, 0.01808215, 0.10018654)]):
            # beige is on the edge, black in the bulk
            pos = position[bool_to_plot_, :]
            ang = traj.angle[bool_to_plot_] % (2 * np.pi)
            ps.draw(pos, ang, scale_factor=scale_factor[ps.size], color=tuple(color))

        ps.draw(position[0], traj.angle[0] % (2 * np.pi), scale_factor=scale_factor[ps.size],
                color=(0, 0.999, 0))  # in is green
        ps.draw(position[-1], traj.angle[-1] % (2 * np.pi), scale_factor=scale_factor[ps.size],
                color=(0, 0, 0.999))  # out is blue
        # mlab.show()
        ps.screenshot(dir=os.path.join(home, 'Analysis', 'bottleneck_c_to_e', 'results', 'images',
                                       traj.filename + str(traj.frames[0]) + '.jpg'))
        mlab.close()
        DEBUG = 1

    @staticmethod
    def create_edge_walked_dict(radius):
        edge_walks = {}
        for size, df_size in df.groupby('size'):
            filenames = df_size['filename']
            ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, name=size + '_' + shape,
                                  geometry=(
                                      'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
            ps.load_space()
            trajs = [get(filename) for filename in filenames]

            for traj in trajs:
                edge_walk = Experiment_Sliding(traj)
                edge_walk.find_boundary(ps, radius)
                edge_walk.find_on_edge(ps, radius)

                # add new_fractions to fraction_on_boundary
                edge_walks[traj.filename] = edge_walk.on_edge.tolist()
        return edge_walks

    @staticmethod
    def create_moving_dict() -> dict:
        moving_dict = {}
        for filename in tqdm(df['filename']):
            traj = get(filename)
            vel = np.linalg.norm(traj.velocity(), axis=1)
            moving = vel > min(max(vel), 1) * 0.2
            print(filename, min(max(vel), 1) * 0.2)
            moving_dict[filename] = moving.tolist()
        return moving_dict

    @staticmethod
    def create_fraction_time_sliding_dict() -> dict:
        df['sliding'] = df['filename'].map(sliding_dict)
        fraction_sliding_dict = {}
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            ew = Experiment_Sliding(traj, sliding=sliding)
            fraction_sliding_dict[filename] = ew.fraction_time(ew.sliding)
        return fraction_sliding_dict

    @staticmethod
    def create_fraction_time_sliding_out_of_moving_dict() -> dict:
        df['sliding'] = df['filename'].map(sliding_dict)
        df['moving'] = df['filename'].map(moving_dict)
        fraction_time_sliding_out_of_moving_dict = {}
        for filename, sliding, moving in tqdm(zip(df['filename'], df['sliding'], df['moving'])):
            traj = get(filename)
            ew = Experiment_Sliding(traj, sliding=sliding, moving=moving)
            fraction_time_sliding_out_of_moving_dict[filename] = np.sum(np.array(sliding).astype(int)) / np.sum(
                np.array(moving).astype(int))
        return fraction_time_sliding_out_of_moving_dict

    @staticmethod
    def create_fraction_distance_walked_sliding_dict() -> tuple:
        df['sliding'] = df['filename'].map(sliding_dict)
        fraction_trans_sliding, fraction_rot_sliding = {}, {}
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            t_sliding, r_sliding = np.sum(trans_dict[filename]), np.sum(rot_dict[filename])
            t_total, r_total = PathLength(traj).translational_distance(), PathLength(traj).rotational_distance()
            fraction_trans_sliding[filename], fraction_rot_sliding[filename] = t_sliding / t_total, r_sliding / r_total
            print(filename, fraction_trans_sliding[filename])
        return fraction_trans_sliding, fraction_rot_sliding

    @staticmethod
    def create_fraction_distance_walked_sliding_trans_and_rot_dict() -> dict:
        df['sliding'] = df['filename'].map(sliding_dict)
        fraction_pL_sliding = {}
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            t_sliding, r_sliding = np.sum(trans_dict[filename]), np.sum(rot_dict[filename])
            t_total, r_total = PathLength(traj).translational_distance(), PathLength(traj).rotational_distance()
            fraction_pL_sliding[filename] = (t_sliding + r_sliding) / (t_total + r_total)
            print(filename, fraction_pL_sliding[filename])
        return fraction_pL_sliding

    @staticmethod
    def time_passed_sliding() -> dict:
        times = {}
        df['sliding'] = df['filename'].map(sliding_dict)
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            my_ew = Experiment_Sliding(traj=traj, sliding=sliding)
            index_succession = my_ew.index_successions(my_ew.sliding)
            times[filename] = [len(indices) / traj.fps for indices in index_succession]
        return times

    @staticmethod
    def pL_passed_sliding() -> tuple:
        trans, rot = {}, {}
        df['sliding'] = df['filename'].map(sliding_dict)
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            my_ew = Experiment_Sliding(traj=traj, sliding=sliding)
            trans[filename], rot[filename] = my_ew.distances_walked(my_ew.sliding)
        return trans, rot

    @staticmethod
    def pL_passed_on_edge() -> tuple:
        trans, rot = {}, {}
        df['sliding'] = df['filename'].map(edge_walks)
        for filename, sliding in tqdm(zip(df['filename'], df['sliding'])):
            traj = get(filename)
            my_ew = Experiment_Sliding(traj=traj, sliding=sliding)
            trans[filename], rot[filename] = my_ew.distances_walked(my_ew.sliding)
        return trans, rot

    @staticmethod
    def plot_correlation_of_time_pL_state(df) -> go.Figure:

        def get_point(traj, indices):
            sliding_window = Sliding_Window(Trajectory_part(traj, VideoChain=[], indices=indices, tracked_frames=[]),
                                            ts=ts_extended[indices[0]:indices[-1]])
            pL = sliding_window.pathLength()
            time = sliding_window.timer()
            prominent_state = sliding_window.prominent_state()
            return [time, pL, prominent_state]

        points = []
        df['sliding'] = df['filename'].map(sliding_dict)
        df['time_series'] = df['filename'].map(time_series_dict)
        for filename, sliding, ts in tqdm(zip(df['filename'], df['sliding'], df['time_series'])):
            print(filename)
            traj = get(filename)
            my_ew = Experiment_Sliding(traj=traj, sliding=sliding)
            index_succession = my_ew.index_successions(my_ew.sliding)

            indices_to_ts_to_frames = np.cumsum([1 / (int(len(traj.frames) / len(ts) * 10) / 10)
                                                 for _ in range(len(traj.frames))]).astype(int)
            ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]

            points += [get_point(traj, indices) for indices in [i for i in index_succession if len(i) > 3*traj.fps]]

        point_colors = {state: [] for state in color_dict.keys()}
        for point in points:
            point_colors[point[-1]].append(point)

        mean = {state: np.array(point_list)[:, 0].astype(float).mean()
                for state, point_list in point_colors.items() if len(point_list) > 0}
        std = {state: np.array(point_list)[:, 0].astype(float).std()
                for state, point_list in point_colors.items() if len(point_list) > 0}

        print(pd.DataFrame.from_dict({'mean': mean, 'std': std}).sort_values('mean'))

        # fig = px.bar(mean)
        # fig = go.Figure()
        # for state, point_list in point_colors.items():
        #     if len(point_list) > 0:
        #         fig.add_scatter(x=np.array(point_list)[:, 0].astype(float), y=np.array(point_list)[:, 1].astype(float),
        #                         mode='markers', marker=dict(color=color_dict[state]), name=state)
        # fig.update_layout(xaxis_title="time [s]", yaxis_title="path length [cm]")
        # return fig

    def clean_exp_from_short_values(self, threshold=5) -> None:
        if self.on_edge is not None and len(self.index_successions(self.on_edge)[0]) > 0:
            new_succesions = [frs for frs in self.index_successions(self.on_edge) if (frs[-1] - frs[0]) > threshold]
            on_edge = np.zeros_like(self.on_edge).astype(bool)
            on_edge[[item for i in new_succesions for item in i]] = True
            self.on_edge = on_edge
        if self.moving is not None and len(self.index_successions(self.on_edge)[0]) > 0:
            new_succesions = [frs for frs in self.index_successions(self.moving) if (frs[-1] - frs[0]) > threshold]
            moving = np.zeros_like(self.moving).astype(bool)
            moving[[item for i in new_succesions for item in i]] = True
            self.moving = moving


if __name__ == '__main__':
    df_ant = pd.concat(dfs_ant.values())
    shape, solver, radius = 'SPT', 'ant', 10
    df = df_ant

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # edge_walks = Experiment_Sliding.create_edge_walked_dict(dfs_ant, radius)
    # with open(results_dir + 'edge_walk.json', 'w') as f:
    #     json.dump(edge_walks, f)
    with open(results_dir + 'edge_walk.json', 'r') as f:
        edge_walks = json.load(f)

    # moving_dict = Experiment_Sliding.create_moving_dict(df_ants)
    # with open(results_dir + 'moving_dict.json', 'w') as f:
    #     json.dump(moving_dict, f)
    with open(results_dir + 'moving_dict.json', 'r') as f:
        moving_dict = json.load(f)

    sliding_dict = {filename: np.logical_and(moving_dict[filename] + [moving_dict[filename][-1]], edge_walks[filename])
                    for filename in df_ant['filename']}

    # ==================== CHECK ====================

    # for filename in filenames:
    filename = 'XL_SPT_4630018_XLSpecialT_1_ants'
    # filenames = df_exp_ant_XL_winner['filename']
    # filename = filenames.iloc[10]
    traj = get(filename)
    # Experiment_Sliding(traj=traj, on_edge=edge_walks[filename]).play_edge_walk()
    # fig = px.line(edge_walks[traj.filename])
    # fig.show()
    moving_frames = [(traj.frames[fr[0]], traj.frames[fr[-1]]) for fr in
                     Experiment_Sliding.index_successions(moving_dict[filename])]
    boundary_frames = [(traj.frames[fr[0]], traj.frames[fr[-1]]) for fr in
                       Experiment_Sliding.index_successions(edge_walks[filename])]

    # ==================== ANALYSIS ====================

    # times_dict = Experiment_Sliding.time_passed_sliding()
    # with open(results_dir + 'times.json', 'w') as f:
    #     json.dump(times_dict, f)
    with open(results_dir + 'times.json', 'r') as f:
        times_dict = json.load(f)

    # trans_dict, rot_dict = Experiment_Sliding.pL_passed_on_edge()
    # with open(results_dir + 'trans_on_edge.json', 'w') as f:
    #     json.dump(trans_dict, f)
    # with open(results_dir + 'rot_on_edge.json', 'w') as f:
    #     json.dump(rot_dict, f)
    with open(results_dir + 'trans_on_edge.json', 'r') as f:
        trans_dict = json.load(f)

    with open(results_dir + 'rot_on_edge.json', 'r') as f:
        rot_dict = json.load(f)

    # fraction_time_sliding_dict = Experiment_Sliding.create_fraction_time_sliding_dict()
    # with open(results_dir + 'fraction_time_sliding_dict.json', 'w') as f:
    #     json.dump(fraction_time_sliding_dict, f)
    with open(results_dir + 'fraction_time_sliding_dict.json', 'r') as f:
        fraction_time_sliding_dict = json.load(f)

    # fraction_time_sliding_out_of_moving_dict = Experiment_Sliding.create_fraction_time_sliding_out_of_moving_dict()
    # with open(results_dir + 'fraction_time_sliding_out_of_moving_dict.json', 'w') as f:
    #     json.dump(fraction_time_sliding_out_of_moving_dict, f)
    with open(results_dir + 'fraction_time_sliding_out_of_moving_dict.json', 'r') as f:
        fraction_time_sliding_out_of_moving_dict = json.load(f)

    # fraction_trans_sliding, fraction_rot_sliding = Experiment_Sliding.create_fraction_distance_walked_sliding_dict()
    # fraction_pL_sliding = Experiment_Sliding.create_fraction_distance_walked_sliding_trans_and_rot_dict()
    # with open(results_dir + 'fraction_trans_sliding.json', 'w') as f:
    #     json.dump(fraction_trans_sliding, f)
    # with open(results_dir + 'fraction_rot_sliding.json', 'w') as f:
    #     json.dump(fraction_rot_sliding, f)
    # with open(results_dir + 'fraction_pL_sliding.json', 'w') as f:
    #     json.dump(fraction_pL_sliding, f)
    with open(results_dir + 'fraction_trans_sliding.json', 'r') as f:
        fraction_trans_sliding = json.load(f)
    with open(results_dir + 'fraction_rot_sliding.json', 'r') as f:
        fraction_rot_sliding = json.load(f)
    with open(results_dir + 'fraction_pL_sliding.json', 'r') as f:
        fraction_pL_sliding = json.load(f)

    # ==================== PLOT ====================

    # fig = Experiment_Sliding.plot_means(fraction_time_sliding_dict)
    # fig.update_yaxes(title_text="Fraction of time spent sliding")
    # fig.update_xaxes(title_text="size")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'fraction_time_sliding')

    # fig = Experiment_Sliding.plot_means(fraction_time_sliding_out_of_moving_dict)
    # fig.update_yaxes(title_text="Fraction of time spent sliding out of moving time")
    # fig.update_xaxes(title_text="size")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'fraction_time_sliding_out_of_moving_dict')

    # fig = Experiment_Sliding.plot_means(fraction_trans_sliding)
    # fig.update_yaxes(title_text="Fraction of translation sliding")
    # fig.update_xaxes(title_text="size")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'fraction_trans_sliding')

    # fig = Experiment_Sliding.plot_means(fraction_rot_sliding)
    # fig.update_yaxes(title_text="Fraction of rotation spent sliding")
    # fig.update_xaxes(title_text="size")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'fraction_rot_sliding')

    # fig = Experiment_Sliding.plot_means(fraction_pL_sliding)
    # fig.update_yaxes(title_text="Fraction of pL sliding")
    # fig.update_xaxes(title_text="size")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'fraction_pL_sliding')

    # fig = Experiment_Sliding.plot_distribution(times_dict)
    # fig.update_xaxes(title_text="time spent sliding [s]")
    # fig.update_yaxes(title_text="frequency")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'time_sliding_distribution')

    pLs = {filename: (np.array(trans_dict[filename]) + np.array(rot_dict[filename])).tolist() for filename in
           df['filename']}
    # fig = Experiment_Sliding.plot_distribution(pLs)
    # fig.update_xaxes(title_text="pL passed sliding [cm]")
    # fig.update_yaxes(title_text="frequency")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'path_sliding_distribution')
    #
    # exit_size = {size: Maze(get(d['filename'].iloc[0])).exit_size for size, d in dfs_ant.items()}
    # exit_size.update({'S': exit_size['Single (1)']})
    # pLs_scaled = {filename: np.array(pLs[filename]) / exit_size[size] for filename, size in
    #               zip(df['filename'], df['size'])}
    # fig = Experiment_Sliding.plot_distribution(pLs_scaled)
    # fig.update_xaxes(title_text="scaled pL passed sliding [exit size]")
    # fig.update_yaxes(title_text="frequency")
    # fig.update_layout(font=dict(family="Times New Roman", size=22))
    # fig.update_layout(width=800, height=600)
    # save_fig(fig, 'scaled_path_on_edge_distribution')

    # for size, df_size in dfs_ant.items():
    #     fig = Experiment_Sliding.plot_bar_chart(df=df_size, block=False)
    #     save_fig(fig, 'ant_' + size, svg=False)

    for size, df_size in dfs_ant.items():
        print(size)
        if size not in ["XL", 'L']:
            fig = Experiment_Sliding.plot_correlation_of_time_pL_state(df=df_size)
            save_fig(fig, 'ant_' + size, svg=False)
