import os

from trajectory_inheritance.get import get
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from DataFrame.import_excel_dfs import *
import seaborn as sns
import numpy as np
from mayavi import mlab
import json
import os
from plotly import express as px
from Analysis.Efficiency.PathLength import PathLength
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from Analysis.PathPy.Path import Path, exp_day
from Directories import network_dir
from DataFrame.plot_dataframe import save_fig
from matplotlib import pyplot as plt
from tqdm import tqdm


class Edge_Walk:
    def __init__(self, traj, boundary=None, on_edge=None):
        self.boundary = boundary
        self.traj = traj
        self.on_edge = on_edge

    def find_boundary_mask(self, ps) -> None:
        """
        Find the mask of the boundary of the maze.
        """
        boundary_and_forbidden = ps.dilate(space=~ps.space, radius=10)
        # self.boundary = np.logical_and(boundary_and_forbidden, ps.space)
        self.boundary = boundary_and_forbidden

        # ps.visualize_space(space=self.boundary)
        # boundary.sum()/boundary.size
        # for traj, color in zip(trajs[:1], colors):
        #     ps.draw(traj.position, traj.angle, scale_factor=0.2, color=tuple(color))

    def fraction_on_boundary(self) -> float:
        """
        Find the fraction of time which the load is in contact with the boundary.
        """
        index_successions = self.index_successions()
        return len(np.concatenate(index_successions))/len(self.traj.frames)

    def distance_walked_on_edge(self) -> tuple:
        """
        Find the distance walked on the edge.
        """
        trans = []
        rot = []

        # Find the consecutive indices of the edge wall
        index_successions = self.index_successions()

        for index_succession in index_successions:
            trans.append(
                PathLength(self.traj).translational_distance(frames=[index_succession[0], index_succession[-1]]))
            rot.append(PathLength(self.traj).rotational_distance(frames=[index_succession[0], index_succession[-1]]))
        return rot, trans

    def fraction_sliding(self) -> float:
        vel = np.linalg.norm(self.traj.velocity(), axis=1)
        vel = np.hstack([vel, vel[-1]])

        sliding = np.sum(vel * np.array(self.on_edge).astype(int)) * 1 / self.traj.fps
        all_motion = np.sum(vel) * 1 / self.traj.fps
        print(self.traj.filename, sliding / all_motion)
        return sliding / all_motion

    def find_on_edge(self, ps) -> None:
        """
        Boolean array whether traj is on the boundary
        """
        on_edge = np.zeros_like(self.traj.angle).astype(bool)
        for i in range(self.traj.position.shape[0]):
            indices = ps.coords_to_indices(x=self.traj.position[i, 0], y=self.traj.position[i, 1],
                                           theta=self.traj.angle[i])
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

    def plot(self):
        index_succession = self.index_successions()
        if self.traj.timer() * len(np.concatenate(index_succession)) / len(traj.frames) / len(index_succession) > 20:
            vel = np.linalg.norm(self.traj.velocity(), axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.traj.frames, y=np.array(self.on_edge).astype(int) * np.mean(vel)))
            fig.add_trace(go.Scatter(x=self.traj.frames, y=vel))
            fig.show()

    def index_successions(self) -> np.array:
        where_on_edge = np.where(self.on_edge)[0]
        index_successions = np.split(where_on_edge, np.where(np.diff(where_on_edge) != 1)[0] + 1)
        return index_successions

    @staticmethod
    def plot_means(dictionary):
        means, std = {}, {}
        for size, df in dfs_ant.items():
            df['to_plot'] = df['filename'].map(dictionary)
            means[size] = df['to_plot'].mean()
            std[size] = df['to_plot'].std()
        fig = px.bar(x=list(means.keys()), y=list(means.values()), error_y=list(std.values()))
        fig.show()

    @staticmethod
    def plot_distribution(dictionary):
        fig = go.Figure()
        for size, t in dictionary.items():
            y_data = np.concatenate(t)
            hist, bins = np.histogram(y_data, bins=100)
            fig.add_trace(go.Scatter(x=bins, y=hist, name=size))
        fig.show()

    @staticmethod
    def plot_bar_chart(df: pd.DataFrame, ax: plt.Axes, block=False):
        with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
            time_series_dict = json.load(json_file)
            json_file.close()

        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\edge_walk.json', 'r') as f:
            edge_walk_dict = json.load(f)

        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\moving_dict.json', 'r') as f:
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
        save_fig(plt.gcf(), 'ant_' + size, svg=False)

    @staticmethod
    def plot_traj_in_cs(ps: ConfigSpace_Maze, trajs: list, bool_to_plot: list):
        """
        Plot the trajectories in the config space.
        """
        num_to_plot = len(trajs)
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        colors = cmap(np.linspace(0.2, 1, num_to_plot))[:, :3]

        ps.visualize_space()
        for traj, color in zip(trajs, colors):
            if bool_to_plot is None:
                bool_to_plot = np.ones_like(traj.angle).astype(bool)
            pos = traj.position[bool_to_plot, :]
            ang = traj.angle[bool_to_plot]
            ps.draw(pos, ang, scale_factor=0.2, color=tuple(color))
        mlab.show()

    @staticmethod
    def create_dict_edge_walked(df: pd.DataFrame):
        edge_walks = {}
        dir = os.getcwd()

        for size, df in df.groupby('size'):
            filenames = df['filename']
            ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, name=size + '_' + shape,
                                  geometry=(
                                      'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
            ps.load_space()
            trajs = [get(filename) for filename in filenames]

            for traj in trajs:
                edge_walk = Edge_Walk(traj)
                edge_walk.find_boundary_mask(ps)
                edge_walk.find_on_edge(ps)

                # add new_fractions to fraction_on_boundary
                edge_walks[traj.filename] = edge_walk.on_edge.tolist()
                fraction_on_boundary[traj.filename] = edge_walk.fraction_on_boundary()

                print(traj.filename, fraction_on_boundary[traj.filename])

            # edge_walk.plot_traj_in_cs(ps=ps, trajs=[trajs[0]], bool_to_plot=[edge_walks[trajs[0].filename]])
        print(os.path.join(dir, 'edge_walk.json'))
        with open(os.path.join(dir, 'edge_walk.json'), 'w') as f:
            json.dump(edge_walks, f)

    @classmethod
    def create_moving_dict(cls, df: pd.DataFrame):
        moving_dict = {}
        for filename in tqdm(df['filename']):
            traj = get(filename)
            vel = np.linalg.norm(traj.velocity(), axis=1)
            moving = vel > 0.2
            moving_dict[filename] = moving.tolist()

        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\moving_dict.json', 'w') as f:
            json.dump(moving_dict, f)

    @classmethod
    def create_fraction_sliding_dict(cls, df: pd.DataFrame):
        # add edge_walk to df
        df['edge walk'] = df['filename'].map(edge_walks)

        fraction_sliding_dict = {}
        for filename, ew in tqdm(zip(df['filename'], df['edge walk'])):
            traj = get(filename)
            fraction_sliding_dict[filename] = Edge_Walk(traj, on_edge=ew).fraction_sliding()

        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\fraction_sliding_dict.json', 'w') as f:
            json.dump(fraction_sliding_dict, f)

    @classmethod
    def create_dicts_distance_walked(cls, df: pd.DataFrame):
        trans, rot = {}, {}

        for filename in df['filenames']:
            traj = get(filename)
            trans[filename], rot[filename] = [], []
            # add new_fractions to fraction_on_boundary
            t, r = Edge_Walk(traj=traj, on_edge=edge_walks[traj.filename]).distance_walked_on_edge()
            trans[filename].append(t)
            rot[filename].append(r)

        # save to json trans and rot
        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\trans.json', 'w') as f:
            json.dump(trans, f)
        with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\rot.json', 'w') as f:
            json.dump(rot, f)

    @classmethod
    def time_between_engagements(cls, df: pd.DataFrame):
        times = {}
        for filename in df['filename']:
            traj = get(filename)
            my_ew = Edge_Walk(traj=traj, on_edge=edge_walks[filename])
            index_succession = my_ew.index_successions()
            times[filename].append(
                traj.timer() * len(np.concatenate(index_succession)) / len(traj.frames) / len(index_succession))


if __name__ == '__main__':
    df_ants = pd.concat(dfs_ant.values())
    shape, solver = 'SPT', 'ant'
    Edge_Walk.create_dict_edge_walked(dfs_ant)

    with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\fraction_on_boundary.json', 'r') as f:
        fraction_on_boundary = json.load(f)
    Edge_Walk.plot_means(fraction_on_boundary)

    Edge_Walk.create_dicts_distance_walked(dfs_ant)
    with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\trans.json', 'r') as f:
        values = json.load(f)
    Edge_Walk.plot_distribution(values)

    filenames = df_exp_ant_XL_winner['filename']
    filename = filenames.iloc[10]
    traj = get(filename)

    with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\edge_walk.json', 'r') as f:
        edge_walks = json.load(f)
    edge_walk = edge_walks[traj.filename]

    fig = px.line(edge_walk)
    fig.show()

    Edge_Walk(traj=traj, on_edge=edge_walk).play_edge_walk()

    Edge_Walk.average_time_between_engagements()
    for size, df in dfs_ant.items():
        fig, ax = plt.subplots()
        Edge_Walk.plot_bar_chart(df=df, ax=ax, block=False)

    Edge_Walk.create_fraction_sliding_dict()

    with open('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\fraction_sliding_dict.json', 'r') as f:
        fraction_on_boundary = json.load(f)
    Edge_Walk.plot_means(fraction_on_boundary)
    DEBUG = 1

# check: XL_SPT_4630014_XLSpecialT_1_ants
