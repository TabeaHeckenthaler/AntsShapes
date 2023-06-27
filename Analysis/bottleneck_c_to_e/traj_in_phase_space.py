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
import os


def calculate_msd(x, y):
    n = len(x)
    msd = np.zeros(n)
    for i in range(1, n):
        dx = x[i:] - x[:-i]
        dy = y[i:] - y[:-i]
        squared_displacements = dx ** 2 + dy ** 2
        msd[i] = np.mean(squared_displacements)
    return msd


def plot_phase_diagrams(frame_indices, winner, extremum, direction, window=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5 * 1))
    ax_vel, ax_coord, ax_angle = axs[0], axs[1], axs[2]

    frames_to_plot = {'black': frame_indices}

    if winner:
        frames_to_plot['red'] = [int(extremum), int(extremum + 6 * traj_original.fps)]
        ax_vel.axvspan(-30, 30, color='lightgreen', alpha=0.5)

    for color, fr_to_plot in frames_to_plot.items():
        s, e = fr_to_plot
        ax_vel.plot(dtheta[s:e], dangle[s:e], color=color)
        if color == 'black':
            ax_vel.plot(dtheta[s], dangle[s], 'o', color='blue', label='start')
            ax_vel.plot(dtheta[e], dangle[e], 'o', color='green', label='end')
        ax_vel.legend()
        ax_vel.set_xlabel('dtheta [degrees / s]')
        ax_vel.set_ylabel('dangle [degrees / s]')

        # center the coordinate system around 0 with the maximum values of the plot
        max_x = {'XL': 30, 'L': 30, 'M': 30, 'S': 30}[size]  # np.max(np.abs(ax[i].get_xlim()))
        max_y = 10  # np.max(np.abs(ax[i].get_ylim()))

        ax_vel.set_xlim(-max_x, max_x)
        ax_vel.set_ylim(-max_y, max_y)

        # draw a line at 0
        ax_vel.axvline(0, color='black')
        ax_vel.axhline(0, color='black')

        ax_coord.plot(xs[s:e], ys[s:e], color=color)
        ax_coord.set_xlabel('x')
        ax_coord.set_ylabel('y')

        # add dot for start and end
        if color == 'black':
            ax_coord.plot(xs[s], ys[s], 'o', color='blue', label='start')
            ax_coord.plot(xs[e], ys[e], 'o', color='green', label='end')

            ax_angle.plot(xs[s], theta[s], 'o', color='blue', label='start')
            ax_angle.plot(xs[e], theta[e], 'o', color='green', label='end')
        ax_coord.legend()

        ax_angle.plot(xs[s:e], theta[s:e], color=color)
        ax_angle.set_xlabel('x')
        ax_angle.set_ylabel('theta')
        ax_coord.set_title(str(traj_original.frames[s]) + ', ' + str(traj_original.frames[e]))

        if window is not None:
            ax_coord.set_xlim(window['x'])
            ax_coord.set_ylim(window['y'])
            ax_angle.set_xlim(window['x'])
            ax_angle.set_ylim(window['theta'])

        ax_angle.legend()

    if size in ['S (> 1)', 'Single (1)']:
        sizename = 'S'
    else:
        sizename = size
    if not os.path.exists(folder + 'c_g_bottleneck_phasespace\\' + sizename):
        os.makedirs(folder + 'c_g_bottleneck_phasespace\\' + sizename)

    fig.savefig(folder + 'c_g_bottleneck_phasespace\\' + sizename + '\\' + filename + str(frame_indices[0]) + '.png')
    plt.close()


def average_velocity_bar_diagram():
    # group into size groups
    size_groups = bottleneck_passing_attempts.groupby('size')
    # plot the average 'vel' for every size
    av_vel, error = {}, {}
    for size, group in size_groups:
        group['vel'] = group['vel'].apply(lambda x: np.array(x))
        av_vel[size] = np.mean(group['vel'].values, axis=0)
        error[size] = np.std(group['vel'].values, axis=0) / np.sqrt(len(group))

    categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    av_vel = {category: av_vel[category] for category in categories}
    error = {category: error[category] for category in categories}
    plt.bar(av_vel.keys(), av_vel.values(), yerr=error.values())
    plt.ylabel('average velocity of direction vector [1/s]')
    plt.tight_layout()
    plt.savefig(folder + 'c_g_bottleneck_phasespace\\' + 'average_velocity_bar_diagram.png')
    plt.close()


def velocity_ratio_plot():
    # group into size groups
    size_groups = bottleneck_passing_attempts.groupby('size')
    # plot the average 'vel' for every size
    av_vel, error = {}, {}
    for size, group in size_groups:
        plt.scatter(group[(group['winner']) & (group['direction'] == 'down')]['dtheta'],
                    group[(group['winner']) & (group['direction'] == 'down')]['dangle'],
                    label='winner: down')
        plt.scatter(group[(group['winner']) & (group['direction'] == 'up')]['dtheta'],
                    group[(group['winner']) & (group['direction'] == 'up')]['dangle'],
                    label='winner: up')
        plt.scatter(group[~group['winner']]['dtheta'], group[~group['winner']]['dangle'], label='not winner')

    plt.legend()
    plt.ylabel('dangle of x and y')
    plt.xlabel('dtheta')
    categories = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    av_vel = {category: av_vel[category] for category in categories}
    error = {category: error[category] for category in categories}
    plt.bar(av_vel.keys(), av_vel.values(), yerr=error.values())
    plt.tight_layout()
    plt.savefig(folder + 'c_g_bottleneck_phasespace\\' + 'average_velocity_bar_diagram.png')
    plt.close()


def plot_msd():
    # get the json file
    with open(folder + 'c_g_bottleneck_phasespace\\' + 'MSD\\' + 'mean_MSD.json', 'r') as json_file:
        msd_dict = json.load(json_file)

    for size, msd_d in msd_dict.items():
        plt.scatter(msd_d['time'][:25], msd_d['msd'][:25], label=size)

    # Plot MSD
    plt.legend()
    plt.xlabel('Time lag [s]')
    plt.ylabel('Mean Square Displacement')
    plt.title('Empirical MSD')
    plt.gcf().savefig(folder + 'c_g_bottleneck_phasespace\\' + 'MSD\\' + 'MSD_sizes.png')
    plt.close()


if __name__ == '__main__':
    folder = 'results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_vel.xlsx',
                                                index_col=0)
    # plot_msd()
    # average_velocity_bar_diagram()
    # velocity_ratio_plot()

    msd_dict = {}
    # load time_series
    with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # split into size groups
    size_groups = bottleneck_passing_attempts.groupby('size')
    for size, group in size_groups:
        # iterate over filenames
        # group = group[group['winner']]

        # # take only 10 first of each size
        # group = group.head(40)
        msd = {}
        for i, row in tqdm(group.iterrows(), total=len(group)):
            filename = row['filename']
            fr_to_plot = row['start'], row['end']
            traj_original = get(filename)

            ts = time_series_dict[filename]
            ts_new = Traj_sep_by_state.extend_time_series_to_match_frames(ts, traj_original)
            traj_original.position[:, 0] = gaussian_filter1d(traj_original.position[:, 0], sigma=5)
            traj_original.position[:, 1] = gaussian_filter1d(traj_original.position[:, 1], sigma=5)
            traj_original.angle = np.unwrap(traj_original.angle)
            traj_original.angle = gaussian_filter1d(traj_original.angle, sigma=5)
            traj_original.angle = traj_original.angle % (2 * np.pi)
            traj_original.ts = ts_new

            # traj = copy(traj_original)
            # traj = traj.cut_off(frame_indices=fr_to_plot)
            # traj.play()

            xs, ys, thetas = traj_original.position[:, 0], traj_original.position[:, 1], traj_original.angle
            theta = np.unwrap(thetas) % (2 * np.pi)
            angle = np.unwrap(np.arctan2(ys, xs))
            dangle = np.diff(angle) * traj_original.fps * 180 / np.pi
            dtheta = np.diff(theta) * traj_original.fps * 180 / np.pi

            # # plot
            plot_phase_diagrams(fr_to_plot, row['winner'], row['extremum'], row['direction'] ,window=None)

            # calculate the average velocity in dangle and dtheta space
            s, e = fr_to_plot
            vel = np.linalg.norm([np.diff(dtheta[s:e]), np.diff(dangle[s:e])], axis=0)

            # add a value to the row
            bottleneck_passing_attempts.loc[i, 'vel'] = np.mean(vel)
            bottleneck_passing_attempts.loc[i, 'dtheta'] = np.mean(dtheta[s:e])
            bottleneck_passing_attempts.loc[i, 'dangle'] = np.mean(dangle[s:e])
            print(filename, bottleneck_passing_attempts.loc[i, 'vel'])

            # Calculate MSD
            msd[filename + str(s)] = calculate_msd(dtheta[s:e:traj_original.fps // 5],
                                                   dangle[s:e:traj_original.fps // 5])

            # # if directory does not exist, create it
        # average over all msd
        seconds = 10
        msd_mean = np.stack([v[:seconds * 5] for v in list(msd.values()) if len(v) >= seconds * 5])
        msd_mean = np.mean(msd_mean, axis=0)
        time = np.arange(0, len(msd_mean), 1) / 5
        msd_dict[size] = {'time': time.tolist(), 'msd': msd_mean.tolist()}

    # save the msd_mean in json
    with open(folder + 'c_g_bottleneck_phasespace\\' + 'MSD\\' + 'mean_MSD.json', 'w') as json_file:
        json.dump(msd_dict, json_file)
        json_file.close()

        # # Plot MSD
        # plt.plot(time, msd_mean)
        # plt.xlabel('Time lag')
        # plt.ylabel('Mean Square Displacement')
        # plt.title('Empirical MSD')
        # plt.gcf().savefig(folder + 'c_g_bottleneck_phasespace\\' + 'MSD\\' + size[0] + '.png')
        # plt.close()

    # save the dataframe
    bottleneck_passing_attempts.to_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts_vel.xlsx')
