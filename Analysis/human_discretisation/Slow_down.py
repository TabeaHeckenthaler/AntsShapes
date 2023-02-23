from trajectory_inheritance.trajectory import Trajectory_part
from trajectory_inheritance.get import get
import json
import numpy as np
from plotly import express as px
import os
import pandas as pd
from Directories import network_dir, home
from ConfigSpace.experiment_sliding import Experiment_Sliding
from DataFrame.gillespie_dataFrame import dfs_gillespie
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from tqdm import tqdm
from DataFrame.import_excel_dfs import find_minimal, dfs_ant, dfs_ant_old
from DataFrame.plot_dataframe import save_fig
import plotly.graph_objects as go
from typing import Union
from mayavi import mlab
from Analysis.Efficiency.PathLength import PathLength


import plotly.figure_factory as ff

columns = ['filename', 'size', 'solver', 'slowdown_mean_speed', 'frames', 'state']


class SlowDown:
    def __init__(self, traj_part):
        self.traj = traj_part
        self.filename = traj_part.filename
        self.frames_of_parent = traj_part.frames_of_parent
        self.states = np.unique(traj_part.states).tolist()

    @staticmethod
    def find_slow_down_indices(traj, slow_vel=0.1, minimal_time=1) -> list:
        vel = traj.velocity()
        speed = np.linalg.norm(vel, axis=1)
        indices_slow = np.where(speed < slow_vel)[0]
        index_successions = np.split(indices_slow, np.where(np.diff(indices_slow) != 1)[0] + 1)
        slow_down_indices = [(i[0], i[-1]) for i in index_successions if len(i) > minimal_time * traj.fps]
        return slow_down_indices

    def to_dict(self):
        return {'filename': self.filename, 'slowdown_mean_speed': self.slowdown_mean_speed(),
                'frames': self.frames_of_parent, 'states': self.states, 'size': self.traj.size,
                'solver': self.traj.solver}

    def slowdown_mean_speed(self):
        speed = np.linalg.norm(traj.velocity(), axis=1)
        return np.mean(speed).astype(float)

    @staticmethod
    def plot_slow_down(traj, slow_down_indices, save=False):
        speed = np.linalg.norm(traj.velocity(), axis=1)
        fig = px.scatter(pd.DataFrame(speed))
        # make background blue between indices a and b on x axis without grey edges
        for a, b in slow_down_indices:
            fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.25)
        # fig.show()
        if save:
            fig.write_image('images\\slow_down_' + traj.filename + '.pdf')


with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

traj = get('large_20210805171741_20210805172610')
ts = time_series_dict[traj.filename]

slow_down_indices = SlowDown.find_slow_down_indices(traj)
SlowDown.plot_slow_down(traj, slow_down_indices, save=True)

for inds in slow_down_indices:
    print(inds)
    traj_part = Trajectory_part(traj, frames=inds, VideoChain=[], tracked_frames=[], states=ts)
    slowDown = SlowDown(traj_part)
    print(slowDown.to_dict())
    DEBUG = 1


DEBUG = 1

