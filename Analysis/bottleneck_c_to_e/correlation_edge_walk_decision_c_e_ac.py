from trajectory_inheritance.trajectory import Trajectory_part
from trajectory_inheritance.get import get
import numpy as np
from ConfigSpace.experiment_sliding import Experiment_Sliding
from DataFrame.gillespie_dataFrame import dfs_gillespie
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
import json
from tqdm import tqdm
from Directories import network_dir
import pandas as pd
import os
from plotly import express as px
from Directories import home
from DataFrame.import_excel_dfs import find_minimal_pL, dfs_ant, dfs_ant_old
from DataFrame.plot_dataframe import save_fig
import plotly.graph_objects as go
from typing import Union
from mayavi import mlab
from Analysis.Efficiency.PathLength import PathLength
from matplotlib import pyplot as plt


import plotly.figure_factory as ff

columns = ['filename', 'frames_of_parent', 'source', 'sink', 'on_edge', 'off_edge', 'fraction_on_edge', 'size',
           'solver']


def extend_time_series_to_match_frames(traj, ts):
    indices_to_ts_to_frames = np.cumsum([1 / (int(len(traj.frames) / len(ts) * 10) / 10)
                                         for _ in range(len(traj.frames))]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


class In_the_bottle:
    def __init__(self, traj_part, ps, ew, radius=10):
        self.traj = traj_part
        self.filename = traj_part.filename
        self.frames_of_parent = traj_part.frames_of_parent
        self.source = self.traj.states[0]
        ind = np.where(np.array(self.traj.states) == 'c')[0][-1]
        if ind == len(self.traj.states) - 1:
            self.sink = None
        else:
            self.sink = self.traj.states[ind + 1]
        self.on_edge, self.off_edge = self.distance_on_off_edge(ps, radius, edge_walk=ew)
        DEBUG = 1

    @staticmethod
    def pL(traj):
        return PathLength(traj).translational_distance(smooth=False) + PathLength(traj).rotational_distance(
            smooth=False)

    @staticmethod
    def cut_traj(traj, ts, buffer=0) -> tuple:
        """
        Divide the trajectory into two lists.
        First, cut off the exp after the last c state.
        Second, split traj into subtrajs that are (1) outside c and cg and (2) within c and cg.
        Return these two lists.
        """
        ts_extended = extend_time_series_to_match_frames(traj, ts)
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
        out_c_trajs = [
            Trajectory_part(traj, indices=[i[0] - buffer * traj.fps, i[-1] + buffer * traj.fps], VideoChain=[],
                            tracked_frames=[], parent_states=ts_extended)
            for i in index_successions_out_cg if len(i) > 0]
        return in_c_trajs, out_c_trajs

    def distance_on_off_edge(self, ps, radius, edge_walk=None) -> tuple:
        on_edge, off_edge = [], []

        if edge_walk is None:
            edge_walk = Experiment_Sliding(self.traj)
        else:
            edge_walk.new_traj(self.traj)
        edge_walk.find_on_edge(ps, radius)
        edge_walk.clean_exp_from_short_values()
        ind_succ = Experiment_Sliding.index_successions(edge_walk.on_edge)

        edge_boolean = np.zeros_like(self.traj.frames).astype(bool)
        edge_boolean[np.hstack(ind_succ)] = True

        where_on_edge = np.array([i for i, x in enumerate(edge_boolean) if x])
        index_successions_on_edge = np.split(where_on_edge, np.where(np.diff(where_on_edge) != 1)[0] + 1)

        where_off_edge = np.array([i for i, x in enumerate(edge_boolean) if not x])
        index_successions_off_edge = np.split(where_off_edge, np.where(np.diff(where_off_edge) != 1)[0] + 1)

        on_edge += [self.pL(Trajectory_part(self.traj, indices=[i[0], i[-1]], VideoChain=[], tracked_frames=[]))
                    for i in index_successions_on_edge if len(i) > 0 and i[-1] - i[0] > 0]
        off_edge += [self.pL(Trajectory_part(self.traj, indices=[i[0], i[-1]], VideoChain=[], tracked_frames=[]))
                     for i in index_successions_off_edge if len(i) > 0 and i[-1] - i[0] > 0]
        return on_edge, off_edge

    def to_dict(self):
        return {'filename': self.filename, 'frames_of_parent': self.frames_of_parent, 'source': self.source,
                'sink': self.sink, 'on_edge': self.on_edge, 'off_edge': self.off_edge,
                'fraction_on_edge': self.fraction_on_edge(), 'solver': self.traj.solver, 'size': self.traj.size}


    def plot_traj_in_cs(self, ps: ConfigSpace_SelectedStates, bool_to_plot: Union[list, None]):
        """
        Plot the trajectories in the config space.
        """
        scale_factor = {'XL': 0.2, 'L': 0.1, 'M': 0.05, 'S': 0.03}
        ps.visualize_space(space=np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg'), reduction=2)
        if bool_to_plot is None:
            bool_to_plot = np.ones_like(self.traj.angle).astype(bool)

        for bool_to_plot_, color in zip([bool_to_plot, np.logical_not(bool_to_plot)],
                                        [(0.96298491, 0.6126247, 0.45145074),
                                         (0.01060815, 0.01808215, 0.10018654)]):
            # beige is on the edge, black in the bulk
            pos = self.traj.position[bool_to_plot_, :]
            ang = self.traj.angle[bool_to_plot_] % (2 * np.pi)
            ps.draw(pos, ang, scale_factor=scale_factor[ps.size], color=tuple(color))

        ps.draw(self.traj.position[0], self.traj.angle[0] % (2 * np.pi), scale_factor=scale_factor[ps.size],
                color=(0, 0.999, 0))  # in is green
        ps.draw(self.traj.position[-1], self.traj.angle[-1] % (2 * np.pi), scale_factor=scale_factor[ps.size],
                color=(0, 0, 0.999))  # out is blue
        # mlab.show()
        ps.screenshot(dir=os.path.join(home, 'Analysis', 'bottleneck_c_to_e', 'results', 'images', self.traj.size + '_' + self.traj.solver,
                                       self.traj.filename + str(self.traj.frames[0]) + '.jpg'))
        mlab.close()
        DEBUG = 1

    def fraction_on_edge(self):
        return sum(self.on_edge) / sum(self.on_edge + self.off_edge)

    @classmethod
    def find_fractions(cls):
        df_results['size_int'] = df_results['size'].map({'XL': 4, 'L': 3, 'M': 2, 'S': 1})
        df_results['on_edge'] = df_results['on_edge'].apply(lambda x: cls.to_list(x))
        df_results['on_edge_sum'] = df_results['on_edge'].apply(lambda x: np.sum(x))
        df_results['minimal path length'] = df_results.apply(find_minimal_pL, axis=1)
        df_results['on_edge_scaled_sum'] = df_results['on_edge_sum'] / df_results['minimal path length']

        df_results.sort_values('size_int', inplace=True)

        df_clean = df_results[(df_results['sink'] != 'cg') & ~(df_results['sink'].isna())]

        percent_solved = {}
        average_fraction = {}
        sem_fractions = {}

        for size, df_clean_size in df_clean.groupby(['size']):
            percent_solved[size] = {sink: len(df_sink) / len(df_clean_size)
                                    for sink, df_sink in df_clean_size.groupby(['sink'])}
            average_fraction[size] = {sink: df_sink['fraction_on_edge'].mean()
                                      for sink, df_sink in df_clean_size.groupby(['sink'])}
            sem_fractions[size] = {sink: df_sink['fraction_on_edge'].std()/len(df_sink['fraction_on_edge'])
                                      for sink, df_sink in df_clean_size.groupby(['sink'])}

        solved_fractions = pd.DataFrame(percent_solved).transpose()
        edge_fractions = pd.DataFrame(average_fraction).transpose()
        sem_fractions = pd.DataFrame(sem_fractions).transpose()

        print('solved_fractions')
        print(solved_fractions)
        print('edge_fractions')
        print(edge_fractions)
        print('sem_fractions')
        print(sem_fractions)


    @classmethod
    def plot_statistics(cls, df_results):
        df_results['on_edge'] = df_results['on_edge'].apply(lambda x: cls.to_list(x))
        df_results['on_edge_sum'] = df_results['on_edge'].apply(lambda x: np.sum(x))
        df_results['minimal path length'] = df_results.apply(find_minimal_pL, axis=1)
        df_results['on_edge_scaled_sum'] = df_results['on_edge_sum'] / df_results['minimal path length']
        marker_dict = {'ac': 'circle', 'e': 'x', 'cg': 'diamond'}
        color_dict = {'XL': 'black', 'L': 'red', 'M': 'blue', 'S': 'green'}

        fig = go.Figure()
        for (sink, size), df in df_results.groupby(['sink', 'size']):
            fig.add_trace(go.Scatter(x=df['fraction_on_edge'],
                                     y=df['on_edge_scaled_sum'],
                                     mode='markers',
                                     marker=dict(size=8, symbol=marker_dict[sink]),
                                     name=sink + ' ' + size,
                                     marker_color=color_dict[size]))
        # add legend to figure
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.show()
        save_fig(fig, 'results\\correlation_edge_walk_decision_c_e_ac')

        colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.95, 0.95, 0.95)]
        y_max = 6
        for (size, sink), df_size_sink in df_results.groupby(['size', 'sink']):
            df = df_size_sink[df_size_sink['on_edge_scaled_sum'] < y_max]
            x = list(df['fraction_on_edge'].values)
            y = list(df['on_edge_scaled_sum'].values.tolist())
            fig = ff.create_2d_density(
                x, y,
                colorscale=colorscale,
                # hist_color='rgb(255, 237, 222)',
                point_size=3
            )

            # change the background color of the histogram
            fig.update_layout(plot_bgcolor='rgb(242, 242, 242)')

            fig.update_layout(title='size ' + size + ' sink ' + sink,
                              xaxis_title='fraction on edge',
                              yaxis_title='on_edge_scaled_sum',
                              )
            # limit x axis in histogram to 0 to 1
            fig.update_xaxes(range=[0, 1])
            # limit y axis to 0 to 15
            # fig.update_yaxes(range=[0, y_max])
            print('sink', sink, 'size', size, 'len', len(df_size_sink))
            save_fig(fig, size + 'sink' + sink)
        DEBUG = 1

    @staticmethod
    def to_list(string: str):
        return [float(x) for x in string.strip('][').split(', ') if len(x) > 0]

    @staticmethod
    def calc(to_do):
        df_shift = pd.read_excel('results\\df_shift.xlsx')
        new_df_results = pd.DataFrame(columns=columns)

        for size, filenames in to_do.items():
            ps = None
            for filename in tqdm(filenames):
                traj = get(filename)
                if ps is None:
                    ps = ConfigSpace_SelectedStates(solver=traj.solver, size=traj.size, shape='SPT',
                                                    geometry=traj.geometry())
                    ps.load_final_labeled_space()

                if traj.solver == 'ant' and \
                        traj.geometry() != ('MazeDimensions_new2021_SPT_ant.xlsx',
                                            'LoadDimensions_new2021_SPT_ant.xlsx'):
                    traj = traj.confine_to_new_dimensions()
                traj.smooth()

                ew = Experiment_Sliding(traj)
                ew.find_on_edge(ps=ps, radius=radius)
                ts = time_series_dict[filename]
                in_c_trajs, out_c_trajs = In_the_bottle.cut_traj(traj, ts, buffer=2)
                # in_c_trajs[0].play()

                for in_c in in_c_trajs:
                    if filename in df_shift['filename'].values:
                        shift_x = df_shift[(df_shift['filename'] == filename)]['shift_x'].values[0]
                        shift_y = df_shift[(df_shift['filename'] == filename)]['shift_y'].values[0]
                        in_c.position = in_c.position - np.array([shift_x, shift_y])

                    in_the_bottle = In_the_bottle(in_c, ps, ew, radius=3)

                    # in_the_bottle.plot_traj_in_cs(ps, ew.on_edge)
                    in_the_bottle.plot_traj_in_2D(bool_to_plot=ew.on_edge)

                    print(in_the_bottle.to_dict()['fraction_on_edge'])
                    print(in_the_bottle.to_dict()['filename'])
                    new_df_results = new_df_results.append(in_the_bottle.to_dict(), ignore_index=True)
                new_df_results.to_excel('results\\new_correlation_results.xlsx')
                DEBUG = 1
        return new_df_results


if __name__ == '__main__':
    solver = 'ant'
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    if solver == 'gillespie':
        df_dict_sep_by_size = dfs_gillespie
        with open(os.path.join(network_dir, 'time_series_selected_states_gillespie.json'), 'r') as json_file:
            time_series_dict = json.load(json_file)
            json_file.close()
        df_results = pd.read_excel(home + '\\Analysis\\bottleneck_c_to_e\\results\\correlation_results_gillespie.xlsx',
                                   usecols=columns)
    elif solver == 'ant':
        df_dict_sep_by_size = {size: pd.concat([dfs_ant[size], dfs_ant_old[size]]) for size in dfs_ant.keys()}
        df_results = pd.read_excel(home + '\\Analysis\\bottleneck_c_to_e\\results\\correlation_results_ant.xlsx',
                                   usecols=columns)
        # 'S_SPT_4710014_SSpecialT_1_ants (part 1)'
        df_results = df_results[~(df_results['filename'].isin(df_dict_sep_by_size['Single (1)']['filename']))]
        DEBUG = 1
    else:
        raise ValueError('solver must be ant or gillespie')
    radius = 10

    # for filename in ['sim_S_20230129-192126New', 'sim_L_20230129-184654New']:
    #     traj = get(filename)
    #     ps = ConfigSpace_SelectedStates(solver=traj.solver, size=traj.size, shape='SPT', geometry=traj.geometry())
    #     ps.load_final_labeled_space()
    #     ew = Experiment_Sliding(traj)
    #
    #     in_cs, out_cs = In_the_bottle.cut_traj(traj, time_series_dict[traj.filename], buffer=2)
    #     for in_c in in_cs:
    #         in_the_bottle = In_the_bottle(in_c, ps, ew, radius=radius)
    #         print(in_the_bottle.to_dict())
    #     DEBUG = 1

    # # read pandas dataframe to save all results
    # # df_results = pd.DataFrame(columns=columns)
    # df_results = df_results[df_results['solver'] != 'gillespie']
    # to_do = {size: [f for f in df_size['filename'].values if f not in df_results['filename'].values]
    #          for size, df_size in df_dict_sep_by_size.items()}

    to_do = {size: df_size['filename'].values for size, df_size in df_dict_sep_by_size.items()}
    #
    # # del to_do['L']
    # # del to_do['M']
    #
    new_df_results = In_the_bottle.calc(to_do)
    # df_results = pd.concat([df_results, new_df_results], ignore_index=True)
    # df_results.to_excel('results\\correlation_results.xlsx')
    #

    # In_the_bottle.plot_statistics(df_results[df_results['filename'].isin(
    #     pd.concat(df_dict_sep_by_size.values())['filename'])])
    # In_the_bottle.find_fractions()
