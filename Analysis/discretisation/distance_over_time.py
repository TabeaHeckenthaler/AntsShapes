from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
import numpy as np
from skfmm import distance  # use this! https://pythonhosted.org/scikit-fmm/
from matplotlib import pyplot as plt
from DataFrame.import_excel_dfs import dfs_human, dfs_ant
from trajectory_inheritance.get import get
from scipy.signal import find_peaks
import pandas as pd
from Setup.Maze import Maze
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
import itertools
from Directories import network_dir
from os import path
from colors import colors_state
import json
from Analysis.PathPy.Path import Path
from scipy import interpolate


class Distance_Finder:
    def __init__(self, cs):
        self.cs = cs
        self.distance = None

    def compute_distances(self, final_label='h'):
        print('computing calc_distances...')
        zero_contour_space = (cs.space_labeled != final_label).astype(int)
        mask = ~np.array(cs.space, dtype=bool)
        zero_contour_space = np.ma.MaskedArray(zero_contour_space, mask)

        dist = distance(zero_contour_space, periodic=(0, 0, 1))
        # in order to mask the 'unreachable' nodes (diagonal or outside of conf_space), set distance there to inf.
        dist[~cs.space] = np.nan
        self.distance = dist
        # plt.imshow(dist[:, :, 0])

    def calc_distance(self, xs, ys, zs):
        coords = [self.cs.coords_to_indices(x, y, z) for x, y, z in zip(xs, ys, zs)]
        distances = np.array([self.distance[coord] for coord in coords])

        # interpolate nans in d
        isnan = np.isnan(distances)
        # calc_distances = np.interp(np.arange(len(calc_distances)), np.flatnonzero(~isnan), calc_distances[~isnan])
        f = interpolate.interp1d(np.arange(len(distances))[~isnan],  distances[~isnan], kind='linear', fill_value='extrapolate')

        # plt.plot(f(range(len(calc_distances))))
        # plt.plot(calc_distances)

        return f(range(len(distances))), distances

    @staticmethod
    def to_list(string: str):
        return [eval(x) for x in string.strip('][').split(', ') if len(x) > 0]

    @classmethod
    def plot_2d_density(cls, df_to_plot, group_type, shade: np.array=None):
        minima = df_to_plot['minima'].apply(lambda x: eval(x)).values
        minima = list(itertools.chain(*minima))
        maxima = df_to_plot['maxima'].apply(lambda x: eval(x)).values
        maxima = list(itertools.chain(*maxima))

        maze = Maze(get(df['filename'].iloc[0]))

        def plotable(coords):
            coords = np.array(coords)
            shift = {'Large C': 1, 'Large NC': 1, 'Small': 0.3, 'S (> 1)': 0.3, 'Single (1)': 0.3}
            coords[:, 0] = coords[:, 0] + shift.get(group_type, 0)
            coords[:, 2] = coords[:, 2] % (2 * np.pi)
            coords[:, 2] = coords[:, 2] * maze.average_radius()
            return coords

        mini_coords, maxi_coords = plotable(minima), plotable(maxima)
        # plot the image of the shade with defined axis scaling and grey colormap
        axim = plt.imshow(shade.T, extent=[0, maze.arena_length, 0, 2 * np.pi * maze.average_radius()], aspect='equal',
                          cmap='Greys', vmin=0, vmax=3)
        # add on top of the image the scatter plot with markersize 2
        axim.axes.scatter(mini_coords[:, 0], mini_coords[:, 2], c='r', s=10, marker='o', alpha=0.5, label='went wrong')
        axim.axes.scatter(maxi_coords[:, 0], maxi_coords[:, 2], c='k', s=10, marker='o', alpha=0.5, label='went right')
        axim.axes.legend()
        axim.axes.set_title(group_type)
        save_fig(plt.gcf(), '2d_scatter_scatter' + group_type)
        plt.close()

    @staticmethod
    def plot_distance(d, time, ts=None, peaks=None, d_real=None):
        # underlay the plot with the states in ts according to the color_state dictionary
        left = 0
        plt.figure()
        if ts is not None:
            dur = Path.time_stamped_series(ts, 1/x.fps)
            for state, duration in dur:
                plt.axvspan(left, left + duration, facecolor=colors_state[state], alpha=0.5, label=state)
                left += duration
            # plt.legend()

        plt.plot(time, d, markersize=0.1, color='#797b8a')
        if d_real is not None:
            plt.plot(time, d_real, markersize=0.1, color='#0f0f0f')
        plt.plot(peaks * 1/x.fps, d[peaks], "x", markersize=10, color="k")

        plt.xlabel('time [s]')
        plt.ylabel('distance')

        plt.savefig('calc_distances' + '\\' + x.filename + '_distance_over_time.png', dpi=300)
        plt.close()

    @staticmethod
    def find_extremata(d, time):
        maxima, properties_max = find_peaks(d, prominence=20)
        minima, properties_min = find_peaks(-d, prominence=20)

        # peaks = np.concatenate((maxima, minima))

        # # Calculate the differences between consecutive peaks
        # peak_diffs = np.diff(peaks)
        # # Calculate the typical length of a peak
        # typical_peak_length = np.mean(peak_diffs)

        # Plot the data and the peaks
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(d)
        # ax.plot(peaks, d[peaks], "x", markersize=10, color="red")
        # save_fig(fig, 'calc_distances' + '\\' + x.filename + '_distance_over_time.png')
        return maxima, minima


if __name__ == '__main__':
    columns = ['filename', 'solver', 'size', 'minima', 'maxima']

    # write dataframe with columns
    # df_results = pd.DataFrame(columns=columns)
    df_results = pd.read_excel('minima_maxima_distances.xlsx', usecols=columns)
    df_results.to_excel('minima_maxima_distances.xlsx')

    # for dfs_solver in [dfs_human, dfs_ant]:
    #     for group_type, df in dfs_solver.items():
    #         print(group_type)
    #         exp = df.iloc[0]
    #
    #         cs = ConfigSpace_SelectedStates(exp['solver'], exp['size'], exp['shape'],
    #                                         (exp['maze dimensions'], exp['load dimensions']))
    #         cs.load_space()
    #         shade = cs.space.any(axis=1)
    #
    #         df_to_plot = df_results[df_results['filename'].isin(df['filename'])]
    #         Distance_Finder.plot_2d_density(df_to_plot, group_type, shade=shade)

    with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    for dfs_solver in [dfs_human, dfs_ant]:
        for key, df in dfs_solver.items():
            exp = df.iloc[0]

            cs = ConfigSpace_SelectedStates(exp['solver'], exp['size'], exp['shape'],
                                            (exp['maze dimensions'], exp['load dimensions']))
            cs.load_final_labeled_space()
            dist_f = Distance_Finder(cs)
            dist_f.compute_distances()

            for filename in df['filename']:
                x = get(filename)
                x.smooth(sec_smooth=1)

                ts = time_series_dict[x.filename]
                ts_extended = Traj_sep_by_state.extend_time_series_to_match_frames(ts, x)

                d_interpolated, d_real = dist_f.calc_distance(x.position[:, 0], x.position[:, 1], x.angle)
                time = np.arange(len(x.frames))/x.fps

                maxima, minima = dist_f.find_extremata(d_interpolated, time)

                dist_f.plot_distance(d_interpolated, time, ts=ts_extended, peaks=np.concatenate((maxima, minima)), d_real=d_real)
                mini = [[x.position[m, 0], x.position[m, 1], x.angle[m]] for m in minima]
                maxi = [[x.position[m, 0], x.position[m, 1], x.angle[m]] for m in maxima]
                new = {'filename': filename, 'solver': x.solver, 'size': x.size, 'minima': mini, 'maxima': maxi}
                df_results = df_results.append(new, ignore_index=True)
            df_results.to_excel('minima_maxima_distances.xlsx')
    #         DEBUG = 1

