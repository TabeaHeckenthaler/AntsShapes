import os.path

from Analysis.bottleneck_c_to_e.correlation_edge_walk_decision_c_e_ac import *
from Setup.Maze import Maze
import cv2
from PhysicsEngine.Display import Display
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from trajectory_inheritance.get import get
from copy import copy
from DataFrame.import_excel_dfs import dfs_ant
from os import path
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
import time


successfull_examples = {'XL': ["XL_SPT_4640021_XLSpecialT_1_ants (part 1): 63047",
                               "XL_SPT_5040012_XLSpecialT_1_ants: 21374",
                               "XL_SPT_4640013_XLSpecialT_1_ants: 15931",
                               "XL_SPT_4630005_XLSpecialT_1_ants (part 1): 21614"],
                        'L': [],
                        'M': [],
                        'S': [],
                        }
#
#
# def shift_coords(xs, ys):
#     # properly shift x and y
#     maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
#                                         'LoadDimensions_new2021_SPT_ant.xlsx'),
#                 size=size, shape='SPT')
#     d = Display('', 1, maze)
#     if key in shifts.index:
#         x_shift = shifts.loc[key, 'x'] / d.ppm
#         y_shift = shifts.loc[key, 'y'] / d.ppm
#
#         xs = [x + x_shift for x in xs]
#         ys = [y + y_shift for y in ys]


def filename_frame(key):
    filename = key.split(':')[0]
    frame = int(key.split(':')[1][1:])
    return filename, frame


def find_consecutive_true_ranges(bool_list):
    consecutive_true_ranges = []
    start_index = None

    for i, value in enumerate(bool_list):
        if value:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                consecutive_true_ranges.append((start_index, i - 1))
                start_index = None

    if start_index is not None:
        consecutive_true_ranges.append((start_index, len(bool_list) - 1))

    return consecutive_true_ranges


def find_traj_parts_in_cubic_subspace(traj, min_distance, window, starting_states=None) -> list:
    """
    Find trajectory parts that pass through a certain window.
    Return the indices (!!) of frames of all the parts, not the frame numbers
    :param traj:
    :param min_distance:
    :param window:
    :return:
    """
    x_boolean = np.logical_and(np.array(traj.position[:, 0]) > window['x'][0],
                               np.array(traj.position[:, 0]) < window['x'][1])
    y_boolean = np.logical_and(np.array(traj.position[:, 1]) > window['y'][0],
                               np.array(traj.position[:, 1]) < window['y'][1])
    theta_boolean = np.logical_and(np.array(traj.angle) > window['theta'][0],
                                   np.array(traj.angle) < window['theta'][1])
    attempt_list = np.logical_and(x_boolean, np.logical_and(y_boolean, theta_boolean))

    # find consecutive True values in attempt_list
    consecutive_true_ranges = find_consecutive_true_ranges(attempt_list)
    # keep ony the ranges that are longer than min_distance
    consecutive_true_ranges = [r for r in consecutive_true_ranges if r[1] - r[0] > min_distance]
    if starting_states is not None:
        consecutive_true_ranges = [r for r in consecutive_true_ranges if traj.ts[r[0]] in starting_states]
    return consecutive_true_ranges


def find_windows(maze):
    # constants that define the windows
    x_c = 0.05  # 0.185  # the larger, the more confined
    y_c1, y_c2 = 4.775, 1.5  # = 4.775, 2.38
    t_c1, t_c2 = 0.9, 2.6

    window_down = {'x': [maze.slits[0] + x_c * np.diff(maze.slits)[0], maze.slits[1] - x_c * np.diff(maze.slits)[0]],
                   'y': [maze.arena_height/y_c1, maze.arena_height/y_c2],
                   'theta': [t_c1, t_c2]}
    window_up = {'x': [maze.slits[0] + x_c * np.diff(maze.slits)[0], maze.slits[1] - x_c * np.diff(maze.slits)[0]],
                 'y': [maze.arena_height - maze.arena_height/y_c2, maze.arena_height - maze.arena_height/y_c1, ],
                 'theta': [2 * np.pi - t_c2, 2 * np.pi - t_c1, ]}
    return window_down, window_up


if __name__ == '__main__':
    folder = 'results\\percentage_around_corner\\'
    # plot_percentages_histogram()
    # plot_statistics_edge_transport()

    # load time_series
    with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # open the shift excel sheet
    shifts = pd.read_excel(folder + 'shifts.xlsx', index_col=0)

    # TODO: Find the attributes of the different sizes that would allow them to remain in the correct area in phasespace

    # # ________________load coordinates _______________________
    for size, df_ant in dfs_ant.items():
        if size in ['S (> 1)', 'Single (1)']:
            sizename = 'S'
        else:
            sizename = size

        # save filename and fr_to_plot in an ecxel sheet
        if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts.xlsx'):
            bottleneck_passing_attempts = pd.DataFrame(columns=['filename', 'size', 'direction', 'start', 'end',
                                                                'winner'])
            bottleneck_passing_attempts.to_excel(
                folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts.xlsx')
            # sizename = 'M'
        # size = 'M'
        # df_ant = dfs_ant[sizename]

        maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                            'LoadDimensions_new2021_SPT_ant.xlsx'),
                    size=sizename, shape='SPT')

        window_down, window_up = find_windows(maze)
        print(window_down)
        print(window_up)

        #
        # # draw the successful movement in dot_theta, and dot_angle of x-y-motion in a plot
        for index, row in tqdm(list(df_ant.iterrows()), desc='size: ' + sizename, total=len(df_ant)):
            filename = row['filename']

            # filename = 'M_SPT_4690001_MSpecialT_1_ants'

            traj_original = get(filename)

            ts = time_series_dict[filename]
            ts_new = Traj_sep_by_state.extend_time_series_to_match_frames(ts, traj_original)
            traj_original.position[:, 0] = gaussian_filter1d(traj_original.position[:, 0], sigma=5)
            traj_original.position[:, 1] = gaussian_filter1d(traj_original.position[:, 1], sigma=5)
            traj_original.angle = np.unwrap(traj_original.angle)
            traj_original.angle = gaussian_filter1d(traj_original.angle, sigma=5)
            traj_original.angle = traj_original.angle % (2 * np.pi)
            traj_original.ts = ts_new

            for window, direction, extr_function in zip([window_down, window_up],
                                                        ['down', 'up'],
                                                        [lambda x: x < 2 * np.pi - 4.27, lambda x: x > 4.27]):

                attempts = find_traj_parts_in_cubic_subspace(traj_original, traj_original.fps * 5, window,
                                                             starting_states=['c'])
                if len(attempts):
                    fig, axs_all = plt.subplots(len(attempts), 3, figsize=(15, 5 * len(attempts)))
                    # plt.show(block=False)
                    if len(attempts) == 1:
                        axs_all = [axs_all]
                    for attempt, axs in zip(attempts, axs_all):
                        s, e = attempt
                        frames_to_plot = {'black': [s, e]}
                        ax_vel, ax_coord, ax_angle = axs[0], axs[1], axs[2]

                        winner = extr_function(traj_original.angle[e])

                        extremum = np.nan
                        if winner:
                            extremum = np.where(extr_function(traj_original.angle[s:]))[0][0] + s

                        for color, fr_to_plot in frames_to_plot.items():
                            traj = copy(traj_original).cut_off(frame_indices=fr_to_plot)
                            # traj.play(wait=5)
                            # traj.adapt_fps(new_fps=traj.fps / (frames[1] - frames[0]))

                            bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                                                 'bottleneck_passing_attempts.xlsx',
                                                                        index_col=0)
                            bottleneck_passing_attempts = bottleneck_passing_attempts.append({'filename': filename,
                                                                                              'size': size,
                                                                                              'direction': direction,
                                                                                              'start': fr_to_plot[0],
                                                                                              'end': fr_to_plot[1],
                                                                                              'winner': winner,
                                                                                              'extremum': extremum
                                                                                              },
                                                                                             ignore_index=True)
                            bottleneck_passing_attempts.to_excel(folder + 'c_g_bottleneck_phasespace\\' +
                                                                 '\\bottleneck_passing_attempts.xlsx')
                            plt.close()

        DEBUG = 1
        #
        #
        #

