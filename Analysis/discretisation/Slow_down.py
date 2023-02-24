from trajectory_inheritance.trajectory import Trajectory_part
from trajectory_inheritance.get import get
import json
import numpy as np
from plotly import express as px
from Analysis.PathPy.Path import color_dict
import plotly.graph_objects as go
import os
import pandas as pd
from Directories import network_dir, home
from Setup.Maze import Maze
from mayavi import mlab
from Analysis.bottleneck_c_to_e.behaviour_in_cg import extend_time_series_to_match_frames
from Setup.MazeFunctions import ConnectAngle
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from DataFrame.plot_dataframe import save_fig
from DataFrame.gillespie_dataFrame import dfs_gillespie
from tqdm import tqdm
from DataFrame.import_excel_dfs import find_minimal, dfs_ant, dfs_ant_old, dfs_human

columns = ['filename', 'size', 'solver', 'mean_speed', 'frames', 'state', 'states', 'time', 'coordinates']
states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}


class SlowDown:
    def __init__(self, traj_part, speed):
        self.traj = traj_part
        self.speed = speed
        self.filename = traj_part.filename
        self.frames_of_parent = traj_part.frames_of_parent
        self.states = np.unique(traj_part.states).tolist()
        if 'g' in self.states:
            DEBUG = 1
        self.state = max(set(self.states), key=self.states.count)

    @staticmethod
    def find_slow_down_indices(traj, slow_vel=None, minimal_time=1) -> list:

        if traj.solver == 'human' and slow_vel is None:
            slow_vel = 0.1
        elif traj.solver == 'ant' and slow_vel is None:
            slow_vel = 0.1

        indices_slow = np.where(traj.speed() < slow_vel)[0]
        index_successions = np.split(indices_slow, np.where(np.diff(indices_slow) != 1)[0] + 1)
        slow_down_indices = [(i[0], i[-1]) for i in index_successions if len(i) > minimal_time * traj.fps]
        return slow_down_indices

    def to_dict(self):
        return {'filename': self.filename, 'mean_speed': self.mean_speed(), 'frames': self.frames_of_parent,
                'states': self.states, 'size': self.traj.size, 'solver': self.traj.solver, 'time': self.seconds(),
                'coordinates': self.coordinates(), 'state': self.state}

    def mean_speed(self):
        return np.mean(self.speed).astype(float)

    def seconds(self):
        return (self.frames_of_parent[1] - self.frames_of_parent[0]) / self.traj.fps

    @staticmethod
    def plot_slow_down(traj, slow_down_indices, save=False):
        speed = np.linalg.norm(traj.velocity(), axis=1)
        speed_df = pd.DataFrame({'speed': speed, 'frames': traj.frames[1:]})
        fig = px.scatter(speed_df, x='frames', y='speed', color_continuous_scale='Viridis')
        # make background blue between indices a and b on x axis without grey edges
        for a, b in slow_down_indices:
            fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.25)
        DEBUG = 1
        # fig.show()
        if save:
            fig.write_image('images\\speed\\slow_down_' + traj.filename + '.pdf')

    def coordinates(self) -> tuple:
        x_mean = np.mean(self.traj.position[:, 0])
        y_mean = np.mean(self.traj.position[:, 1])
        theta = np.mean(ConnectAngle(self.traj.angle, self.traj.shape)) % (np.pi * 2)
        return x_mean, y_mean, theta

    def plot_in_cs(self, cs: ConfigSpace_Maze, fig=None, save=False):
        scale_factor = {'XL': 0.2, 'L': 0.1, 'M': 0.05, 'S': 0.03, 'Large': 1., 'Medium': 0.5, 'Small Far': 0.2,
                        'Small Near': 0.2, 'Small': 0.2}
        cs.draw(self.traj.position, self.traj.angle, scale_factor=scale_factor[cs.size])

    @classmethod
    def calc(cls, df):
        new_results = pd.DataFrame(columns=columns)

        # cs = ConfigSpace_Maze(solver=df['solver'].iloc[0], size=df['size'].iloc[0], shape='SPT',
        #                       geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

        for filename in tqdm(df['filename']):
            # filename = 'large_20210805171741_20210805172610'
            x = get(filename)
            speed_x = x.speed()
            ts = time_series_dict[x.filename]
            ts_extended = extend_time_series_to_match_frames(ts, x)
            # fig = cs.new_fig()
            # cs.visualize_space(fig=fig)

            slow_down_indices = SlowDown.find_slow_down_indices(x)
            # SlowDown.plot_slow_down(x, slow_down_indices, save=True)

            for inds in slow_down_indices:
                traj_part = Trajectory_part(x, frames=inds, VideoChain=[], tracked_frames=[], states=ts_extended)
                slowDown = SlowDown(traj_part, speed=speed_x[inds[0]:inds[1]])
                d = slowDown.to_dict()
                new_results = new_results.append(d, ignore_index=True)
                print(slowDown.to_dict())
                # slowDown.plot_in_cs(cs, fig=fig)
            # cs.screenshot(dir=os.path.join(home, 'Analysis', 'human_discretisation', 'images',
            #                                x.size, x.filename + '.jpg'))
            # mlab.close()
        return new_results

    @staticmethod
    def plot_correlation():
        size_dict = {'Large': 8, 'Medium': 6, 'Small Far': 4, 'Small Near': 4, 'Small': 4}
        fig = go.Figure()
        for (state, size), df in results.groupby(['state', 'size']):
            fig.add_trace(go.Scatter(x=df['mean_speed'],
                                     y=df['time'],
                                     mode='markers',
                                     marker=dict(size=size_dict[size], symbol='circle'),
                                     name=state + ' ' + size,
                                     marker_color=color_dict[eval(state)]))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_xaxes(title_text="mean speed [m/s]")
        fig.update_yaxes(title_text="time [s]")

        fig.show()

        DEBUG = 1

    @staticmethod
    def plot_correlation1():
        size_dict = {'Large': 8, 'Medium': 6, 'Small Far': 4, 'Small Near': 4, 'Small': 4}
        fig = go.Figure()
        for (state, size), df in results.groupby(['state', 'size']):
            fig.add_trace(go.Scatter(x=df['size'],
                                     y=df['time'],
                                     mode='markers',
                                     marker=dict(size=size_dict[size], symbol='circle'),
                                     name=state + ' ' + size,
                                     marker_color=color_dict[eval(state)]))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_xaxes(title_text="size")
        fig.update_yaxes(title_text="time [s]")

        fig.show()

    @staticmethod
    def to_list(string: str):
        return [eval(x) for x in string.strip('][').split(', ') if len(x) > 0]

    @staticmethod
    def number_of_slow_downs(filename='small_20220907175655_20220907175907'):
        sldws = results[results['filename'] == filename]
        counter = {state: sldws['state'].apply(lambda x: ''.join(c for c in repr(x) if c.isalpha()
                                                                 or c.isnumeric()) == state).sum() for state in states}
        counter['b1'] = counter['b1'] + counter['b2']
        del counter['b2']
        return counter

    @staticmethod
    def plot_mean_number_of_slowdowns():
        means = {}
        sem = {}
        for group_type, df in dfs_human.items():
            numbers = {}
            df['N_s'] = df['filename'].apply(lambda x: SlowDown.number_of_slow_downs(x))
            for state in [state for state in states if state != 'b2']:
                numbers[state] = [d[state] for d in df['N_s']]
            means[group_type] = {state: np.mean(numbers[state]) for state in states if state != 'b2'}
            sem[group_type] = {state: np.std(numbers[state])/len(numbers[state]) for state in states if state != 'b2'}

        # plot the means and std of the number of slow downs per state per group type
        fig = go.Figure()
        for group_type in dfs_human.keys():
            fig.add_trace(go.Bar(x=list(means[group_type].keys()), y=list(means[group_type].values()),
                                 name=group_type, error_y=dict(type='data', array=list(sem[group_type].values()))))
        fig.update_layout(barmode='group')
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_xaxes(title_text="state")
        fig.update_yaxes(title_text="number of slow downs")
        fig.show()
        save_fig(fig, 'number_of_slow_downs.pdf')

    @staticmethod
    def find_coords_of_all_slowdowns(filename):
        sldws = results[results['filename'] == filename]
        coords = []
        for index, row in sldws.iterrows():
            coords.append(eval(row['coordinates']))
        return coords

    @staticmethod
    def plot_2d_density(df):
        df['coords'] = df['filename'].apply(lambda x: SlowDown.find_coords_of_all_slowdowns(x))
        coords = np.vstack(df['coords'])
        maze = Maze(get(df['filename'].iloc[0]))
        coords[:, 2] = coords[:, 2] * maze.average_radius()

        x = coords[:, 0]
        y = coords[:, 2]
        trace_scatter = go.Scatter(x=x, y=y, mode='markers')
        trace_density = go.Densitymapbox(
            lat=y, lon=x, z=[1] * len(x),  # use a constant value for all points
            radius=25
        )
        layout = go.Layout(
            title=group_type,
            width=800,  # set the width of the plot to 800 pixels
            height=1200,  # set the height of the plot to 800 pixel
            mapbox=dict(
                center=dict(lat=np.mean(y), lon=np.mean(x)),
                style='white-bg',
                zoom=4),
            paper_bgcolor='rgba(0,0,0,0)',  # make the plot background transparent
            plot_bgcolor='rgba(0,0,0,0)'  # make the plot background transparent
        )
        fig = go.Figure(data=[trace_density, trace_scatter], layout=layout)
        fig.show()
        # fig.write_image('2d_scatter_density.svg')
        save_fig(fig, '2d_scatter_density' + group_type)


if __name__ == '__main__':
    # create a pandas dataframe with all the slowdowns

    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # results = pd.DataFrame(columns=columns)
    # for group_type, df in dfs_ant.items():
    #     # if group_type == 'L':
    #     new_results = SlowDown.calc(df)
    #     results = results.append(new_results, ignore_index=True)
    #     results.to_excel(os.path.join(home, 'Analysis', 'discretisation', 'slow_down_ant.xlsx'))
    #
    # for group_type, df in dfs_ant_old.items():
    #     new_results = SlowDown.calc(df)
    #     results = results.append(new_results, ignore_index=True)
    #     results.to_excel(os.path.join(home, 'Analysis', 'discretisation', 'slow_down_ant.xlsx'))

    # dfs = dfs_human
    # dfs = {size: pd.concat([dfs_ant[size], dfs_ant_old[size]]) for size in dfs_ant.keys()}
    dfs = dfs_ant

    results = pd.read_excel(os.path.join(home, 'Analysis', 'discretisation', 'slow_down_ant.xlsx'))
    # plot the means and std of the number of slow downs per state per group type
    # SlowDown.plot_mean_number_of_slowdowns()
    for group_type, df in dfs.items():
        SlowDown.plot_2d_density(df)
    DEBUG = 1
    # SlowDown.plot_correlation1()

    """
    Goal: Show that humans divide Configuration space into discrete states (nodes). 
    We can identify where the nodes are, by assuming that at every node a new decision has to be reached. 
    Decision making slows down the movement.    
    We can identify the nodes by looking at the time series of the states.
    
    What kind of graphs would be useful here?
    How many slow downs per experiment per state?
    
    bar plot of group_types on the x axis. 
    Every group_type then has bars for every state. 
    The height of the bar corresponds to the number of slow downs per experiment per state.    
    """
    DEBUG = 1

