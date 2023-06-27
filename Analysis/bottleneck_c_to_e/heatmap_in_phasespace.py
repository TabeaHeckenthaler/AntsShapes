import pandas as pd
from trajectory_inheritance.get import get
from os import path
import json
from Directories import network_dir
from trajectory_inheritance.trajectory_sep_into_states import Traj_sep_by_state
from scipy.ndimage import gaussian_filter1d
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from Setup.Maze import Maze
import cv2
import os
from matplotlib import cm


def save_coord_dict():
    coord_dict = {}
    size_groups = bottleneck_passing_attempts.groupby('size')
    for size, group in size_groups:
        # group = group.head(1)
        for i, row in tqdm(group.iterrows(), desc=size, total=len(group)):
            filename = row['filename']
            fr_to_plot = row['start'], row['end']
            traj_original = get(filename)

            traj_original.position[:, 0] = gaussian_filter1d(traj_original.position[:, 0], sigma=5)
            traj_original.position[:, 1] = gaussian_filter1d(traj_original.position[:, 1], sigma=5)
            traj_original.angle = np.unwrap(traj_original.angle)
            traj_original.angle = gaussian_filter1d(traj_original.angle, sigma=5)
            traj_original.angle = traj_original.angle % (2 * np.pi)

            x, y, theta = traj_original.position[row['start']:row['end'], 0].tolist(), traj_original.position[row['start']:row['end'], 1].tolist(), \
                traj_original.angle[row['start']:row['end']].tolist()
            new_fps = 5
            # save only first 3 digits after the decimal point
            x = [x[i] for i in range(0, len(x), int(traj_original.fps / new_fps))]
            x = [round(xi, 3) for xi in x]
            y = [y[i] for i in range(0, len(y), int(traj_original.fps / new_fps))]
            y = [round(yi, 3) for yi in y]
            theta = [theta[i] for i in range(0, len(theta), int(traj_original.fps / new_fps))]
            theta = [round(thetai, 5) for thetai in theta]

            coord_dict[filename + '_' + str(fr_to_plot[0])] = {'x': x, 'y': y, 'theta': theta, 'fps': new_fps}

    # save coord_dict
    with open(folder + 'c_g_bottleneck_phasespace\\coord_dict.json', 'w') as json_file:
        json.dump(coord_dict, json_file)
        json_file.close()


def plot_phase_diagrams(x, y, theta, fps, frame_indices, winner, extremum):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5 * 1))
    ax_vel, ax_coord, ax_angle = axs[0], axs[1], axs[2]

    frames_to_plot = {'black': frame_indices}

    if winner:
        ax_vel.axvspan(-50, 50, color='lightgreen', alpha=0.5)

    for color, fr_to_plot in frames_to_plot.items():
        # ax_vel.plot(dtheta, dangle, color=color)
        # ax_vel.plot(dtheta[0], dangle[0], 'o', color='yellow', label='start')
        # ax_vel.plot(dtheta[-1], dangle[-1], 'o', color='green', label='end')
        # ax_vel.legend()
        # ax_vel.set_xlabel('dtheta [degrees / s]')
        # ax_vel.set_ylabel('dangle [degrees / s]')

        # center the coordinate system around 0 with the maximum values of the plot
        if size in ['S (> 1)', 'Single (1)']:
            max_x = {'XL': 50, 'L': 50, 'M': 50, 'S': 50}['S']  # np.max(np.abs(ax[i].get_xlim()))
        else:
            max_x = {'XL': 50, 'L': 50, 'M': 50, 'S': 50}[size]  # np.max(np.abs(ax[i].get_xlim()))
        max_y = 10  # np.max(np.abs(ax[i].get_ylim()))

        ax_vel.set_xlim(-max_x, max_x)
        ax_vel.set_ylim(-max_y, max_y)

        # draw a line at 0
        ax_vel.axvline(0, color='black')
        ax_vel.axhline(0, color='black')

        # add as a background the corresponding heatmap
        # load the .npy file

        if size == 'S (> 1)':
            size_new = 'SMany'
        elif size == 'Single (1)':
            size_new = 'SSingle'
        else:
            size_new = size

        heatmap0 = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'heatmap_0.npy')
        y_edges = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'yedges.npy')
        x_edges = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'xedges.npy')

        # plot heatmap as background, with scale from 0 to 1
        ax_coord.imshow(heatmap0, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='cool', alpha=1,
                        vmin=0, vmax=1)

        ax_coord.plot(x, y, color=color)
        ax_coord.set_xlabel('x')
        ax_coord.set_ylabel('y')

        ax_coord.plot(x[0], y[0], 'o', color='yellow', label='start')
        ax_coord.plot(x[-1], y[-1], 'o', color='green', label='end')

        ax_angle.plot(x[0], theta[0], 'o', color='yellow', label='start')
        ax_angle.plot(x[-1], theta[-1], 'o', color='green', label='end')
        ax_coord.legend()

        # add as a background the corresponding heatmap
        # load the .npy file
        heatmap1 = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'heatmap_1.npy')
        theta_edges = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'thetaedges.npy')
        x_edges = np.load(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size_new + 'xedges.npy')

        # plot heatmap as background, with scale from 0 to 1
        ax_angle.imshow(heatmap0, extent=[x_edges[0], x_edges[-1], theta_edges[0], theta_edges[-1]], cmap='cool', alpha=1,
                        vmin=0, vmax=1)

        ax_angle.plot(x, theta, color=color)
        ax_angle.set_xlabel('x')
        ax_angle.set_ylabel('theta')
        # ax_coord.set_title(str(traj_original.frames[s]) + ', ' + str(traj_original.frames[e]))

        ax_angle.legend()

    if size in ['S (> 1)', 'Single (1)']:
        sizename = 'S'
    else:
        sizename = size
    if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\' + sizename):
        os.makedirs(folder + 'c_g_bottleneck_phasespace\\' + sizename)

    fig.savefig(folder + 'c_g_bottleneck_phasespace\\' + sizename + '\\' + filename + str(frame_indices[0]) + '.png')
    plt.close()


def find_windows(maze):
    # constants that define the windows
    x_c = 0.05  # 0.185  # the larger, the more confined
    y_c1, y_c2 = 4.775, 1.5  # = 4.775, 2.38
    t_c1, t_c2 = 0.9, 2.6

    window_down = {'x': [maze.slits[0] + x_c * np.diff(maze.slits)[0], maze.slits[1] - x_c * np.diff(maze.slits)[0]],
                   'y': [maze.arena_height / y_c1, maze.arena_height / y_c2],
                   'theta': [t_c1, t_c2]}
    window_up = {'x': [maze.slits[0] + x_c * np.diff(maze.slits)[0], maze.slits[1] - x_c * np.diff(maze.slits)[0]],
                 'y': [maze.arena_height - maze.arena_height / y_c2, maze.arena_height - maze.arena_height / y_c1, ],
                 'theta': [2 * np.pi - t_c2, 2 * np.pi - t_c1, ]}
    return window_down, window_up


def create_heatmap(x_lists, y_lists, theta_lists, size, color='blue', name=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Heatmap of passing attempts around bottleneck')

    if size in ['S (> 1)', 'Single (1)']:
        sizename = 'S'
    else:
        sizename = size
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=sizename, shape='SPT')
    window_down, window_up = find_windows(maze)

    # plot all the trajectories in x_lists and y_lists
    for i in range(len(x_lists)):
        ax.plot(x_lists[i], y_lists[i], alpha=0.5, color=color)
    ax.set_xlim(window_down['x'][0], window_down['x'][1])
    ax.set_ylim(window_down['y'][0], window_up['y'][1])

    # # heatmap with x_lists and y_lists
    # heatmap_0, xedges, yedges = np.histogram2d(np.hstack(x_lists), np.hstack(y_lists), bins=100,
    #                                            range=[[window_down['x'][0], window_down['x'][1]],
    #                                                   [window_down['y'][0], window_up['y'][1]]], normed=True
    #                                            )
    # # heatmap_0 = heatmap_0.astype(bool)
    # heatmap_0 = heatmap_0.T
    # ax.imshow(heatmap_0, cmap='binary', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # ax.equal_aspect = True

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_xlabel('x')
    ax1.set_ylabel('theta')

    # plot all the trajectories in x_lists and y_lists
    for i in range(len(x_lists)):
        ax1.plot(x_lists[i], theta_lists[i], alpha=0.5, color=color)
    ax1.set_xlim(window_down['x'][0], window_down['x'][1])
    ax1.set_ylim(window_down['theta'][0], window_up['theta'][1])

    # heatmap with x_lists and y_lists
    # heatmap_1, xedges, thetaedges = np.histogram2d(np.hstack(x_lists), np.hstack(theta_lists), bins=100,
    #                                                range=[[window_down['x'][0], window_down['x'][1]],
    #                                                       [window_down['theta'][0], window_up['theta'][1]]], normed=True
    #                                                )
    # # heatmap_1 = heatmap_1.astype(bool)
    # heatmap_1 = heatmap_1.T
    # ax1.imshow(heatmap_1, cmap='binary', interpolation='nearest',
    #            extent=[xedges[0], xedges[-1], thetaedges[0], thetaedges[-1]])
    # ax1.equal_aspect = True

    # save fig and fig1
    if size == 'S (> 1)':
        size = 'SMany'
    elif size == 'Single (1)':
        size = 'SSingle'
    folder = 'results\\percentage_around_corner\\'
    if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\heatmaps'):
        os.makedirs(folder + 'c_g_bottleneck_phasespace\\heatmaps')
    fig.savefig(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_xy_' + name + '.png')
    fig1.savefig(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_xtheta_' + name + '.png')
    plt.close()

    # # save the heatmaps and the edges
    # folder = 'results\\percentage_around_corner\\'
    # if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\heatmaps'):
    #     os.makedirs(folder + 'c_g_bottleneck_phasespace\\heatmaps')
    # np.save(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_0.npy', heatmap_0)
    # np.save(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_1.npy', heatmap_1)
    # np.save(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'xedges.npy', xedges)
    # np.save(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'yedges.npy', yedges)
    # np.save(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'thetaedges.npy', thetaedges)


def save_all_heatmaps():
    # split into size groups
    for size in [ 'M', 'XL', 'L','S (> 1)', 'Single (1)']:
        group = bottleneck_passing_attempts[bottleneck_passing_attempts['size'] == size]
        # split into lists of winners and non winners
        winners = group[group['winner']]
        non_winners = group[~group['winner']]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.set_xlabel('x')
        ax1.set_ylabel('theta')

        if size in ['S (> 1)', 'Single (1)']:
            sizename = 'S'
        else:
            sizename = size
        maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                            'LoadDimensions_new2021_SPT_ant.xlsx'),
                    size=sizename, shape='SPT')
        window_down, window_up = find_windows(maze)
        ax.set_xlim(window_down['x'][0], window_down['x'][1])
        ax.set_ylim(window_down['y'][0], window_up['y'][1])
        ax1.set_xlim(window_down['x'][0], window_down['x'][1])
        ax1.set_ylim(window_down['theta'][0], window_up['theta'][1])
        ax1.set_aspect(maze.average_radius())

        for gr, color in zip([non_winners, winners, ], ['red', 'green', ]):
            keys = [filename + '_' + str(int(start_frame)) for filename, start_frame in
                    zip(gr['filename'], gr['start'])]
            x_lists = [coord_dict[key]['x'] for key in keys]
            y_lists = [coord_dict[key]['y'] for key in keys]
            theta_lists = [coord_dict[key]['theta'] for key in keys]

            plt.show(block=False)

            for i in range(len(x_lists)):
                ax.plot(x_lists[i], y_lists[i], alpha=0.3, color=color)
                ax1.plot(x_lists[i], theta_lists[i], alpha=0.3, color=color)
                DEBUG = 1

        if size == 'S (> 1)':
            size = 'SMany'
        elif size == 'Single (1)':
            size = 'SSingle'
        folder = 'results\\percentage_around_corner\\'
        if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\heatmaps'):
            os.makedirs(folder + 'c_g_bottleneck_phasespace\\heatmaps')
        fig.savefig(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_xy' + '.png')
        fig1.savefig(folder + 'c_g_bottleneck_phasespace\\heatmaps\\' + size + 'heatmap_xtheta_' + '.png')
        plt.close()

        # create_heatmap(x_lists, y_lists, theta_lists, size, color=color, name=color)


def plot_vectors(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the number of vectors
    num_vectors = len(vectors)

    # Create a colormap
    cmap = cm.get_cmap('rainbow')

    # Plot each vector
    for i, vector in enumerate(vectors):
        x, y, z = vector

        # Plot the vector
        ax.quiver(0, 0, 0, x, y, z, color=cmap(i / num_vectors))

    # Set plot limits and labels
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('theta')

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    fig.colorbar(sm)
    # save figure
    folder = 'results\\percentage_around_corner\\'
    if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\vectors'):
        os.makedirs(folder + 'c_g_bottleneck_phasespace\\vectors')
    fig.savefig(folder + 'c_g_bottleneck_phasespace\\vectors\\' + filename + str(fr_to_plot[0]) + 'vectors.png')
    plt.close()


if __name__ == '__main__':

    folder = 'results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_messed_up.xlsx',
                                                index_col=0)
    save_coord_dict()
    with open(folder + 'c_g_bottleneck_phasespace\\coord_dict.json', 'r') as json_file:
        coord_dict = json.load(json_file)
    save_all_heatmaps()

    # split into size groups
    size_groups = bottleneck_passing_attempts.groupby('size')
    for size, group in size_groups:
        group = group.head(3)
        for i, row in tqdm(group.iterrows(), total=len(group)):
            filename = row['filename']
            fr_to_plot = [int(row['start']), int(row['end'])]

            x = get(filename)

            key = filename + '_' + str(fr_to_plot[0])
            if key in coord_dict:
                x, y, theta = coord_dict[key]['x'], coord_dict[key]['y'], coord_dict[key]['theta']
            else:
                raise Exception('key not found')

            theta = np.unwrap(theta) % (2 * np.pi)
            x_diff = np.diff(x) * 10
            y_diff = np.diff(y) * 10
            theta_diff = np.diff(theta) * 10

            x_acc = np.diff(x_diff) * 10
            y_acc = np.diff(y_diff) * 10
            theta_acc = np.diff(theta_diff) * 10

            print(filename)
            print(size)
            print(np.mean(np.linalg.norm(np.array([x_acc, y_acc, theta_acc]).T, axis=1)))

            plot_vectors(np.array([x_diff, y_diff, theta_diff]).T)

            # fig = plt.figure()
            # azimuth_angle = np.unwrap(np.arctan2(y_diff, x_diff))
            # polar_angle = np.unwrap(np.arctan2(np.sqrt(x_diff ** 2 + y_diff ** 2), theta_diff))
            # plt.plot(np.diff(azimuth_angle), marker='o')
            # plt.plot(np.diff(polar_angle), marker='o')

            # plot_phase_diagrams(x, y, theta, 10, fr_to_plot, row['winner'], row['extremum'])

    # save coord_dict
    with open(folder + 'c_g_bottleneck_phasespace\\coord_dict.json', 'w') as json_file:
        json.dump(coord_dict, json_file)
        json_file.close()
