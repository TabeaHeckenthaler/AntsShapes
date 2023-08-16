import numpy as np
from tqdm import tqdm
import pandas as pd
from Directories import home, network_dir
import json
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import os
from matplotlib import cm as mpl
from Setup.Maze import Maze
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from scipy.ndimage import distance_transform_edt


centerOfMass_shift = - 0.08
SPT_ratio = 2.44 / 4.82

with open(home + '\\ConfigSpace\\time_series_ant.json', 'r') as json_file:
    time_series = json.load(json_file)
    json_file.close()


sizes_per_solver = {'ant': ['L', 'M', 'S (> 1)', 'XL', ],
                    'sim': ['S', 'M', 'L', 'XL'],
                    # 'sim': ['XS', 'S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium C', 'Large C', 'Medium NC', 'Large NC'],
                    }

date = 'SimTrjs_RemoveAntsNearWall=False'
date1 = 'SimTrjs_RemoveAntsNearWall=True'
#
df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
df_gillespie1 = pd.read_excel(home + '\\Gillespie\\' + date1 + '_sim.xlsx')
df_human = pd.read_excel(home + '\\DataFrame\\final\\df_human.xlsx')
df_ant_excluded = pd.read_excel(home + '\\DataFrame\\final\\df_ant_excluded.xlsx')

slits = {'XL': [13.02, 18.85], 'L': [6.42, 9.53], 'M': [3.23, 4.79], 'S': [2.65, 3.42], 'S (> 1)': [2.65, 3.42]}
arena_height = {'XL': 19.1, 'L': 9.5, 'M': 4.8, 'S': 2.48, 'S (> 1)': 2.48}
exit_size = {'XL': 3.75, 'L': 1.85, 'M': 0.92, 'S': 0.54, 'S (> 1)': 0.54}

dfs = {'ant': df_ant_excluded,
       'SimTrjs_RemoveAntsNearWall=True_sim': df_gillespie1,
       'SimTrjs_RemoveAntsNearWall=False_sim': df_gillespie,
       'human': df_human}

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(home + '\\Gillespie\\' + date1 + '_sim_time_series.json', 'r') as json_file:
    time_series_dict.update(json.load(json_file))
    json_file.close()


def calc_back_corner_positions(x, maze):
    # find frames, where only one edge_locations is behind the first slit
    [shape_height, shape_width, shape_thickness, short_edge] = maze.getLoadDim()

    h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
    corners = np.array([[-shape_width / 2 - h, -shape_height / 2],
                        [-shape_width / 2 - h, shape_height / 2]])

    # find the position of the two corners of the shape in every frame
    corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
    corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])
    return corner1, corner2


def calc_front_corner_positions(x, maze):
    # find frames, where only one edge_locations is behind the first slit
    [shape_height, shape_width, shape_thickness, short_edge] = maze.getLoadDim()

    h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
    corners = np.array([[shape_width / 2 - shape_thickness - h, shape_height / 2 * SPT_ratio],
                        [shape_width / 2 - h, shape_height / 2 * SPT_ratio]])

    # find the position of the two corners of the shape in every frame
    corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
    corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])
    return corner1, corner2


def find_backroom_frames(traj):
    maze = Maze(traj)
    traj.smooth(sec_smooth=1 / traj.fps * 20)
    backcorners = calc_back_corner_positions(traj, maze)
    frontcorners = calc_front_corner_positions(traj, maze)

    # find all frames of the trajectory, where all corners behind maze.slits[0]
    frames = np.where((backcorners[0][0, :] < maze.slits[0] * 1.3) &
                      (backcorners[1][0, :] < maze.slits[0] * 1.3) &
                      (frontcorners[0][0, :] < maze.slits[0] * 1.3) &
                      (frontcorners[1][0, :] < maze.slits[0] * 1.3))[0]

    # find succesions of frames
    succesions = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
    succesions = [[suc[0], suc[-1]] for suc in succesions if len(suc) > traj.fps * 10]

    # only include those that exit b and enter c
    succesions = [suc for suc in succesions if (suc[0] > 0) and
                  np.max([frontcorners[0][0, suc[0] - 1],
                          frontcorners[1][0, suc[0] - 1]]) > maze.slits[0] and
                  traj.position[suc[0], 0] < maze.slits[0] * 1.05]

    # succesions = [suc for suc in succesions if (suc[-1] < len(traj.frames) - 1) and
    #               np.max([backcorners[0][0, suc[-1] + 1],
    #                       backcorners[1][0, suc[-1] + 1]]) > maze.slits[0]]

    succesions = [suc for suc in succesions if (suc[-1] < len(traj.frames) - 1) and
                  np.max([frontcorners[0][0, suc[-1] + 1],
                          frontcorners[1][0, suc[-1] + 1]]) > maze.slits[0]]

    trajs = []
    for suc in succesions:
        traj_backroom = traj.cut_off([suc[0], suc[-1]])
        trajs.append(traj_backroom)
    return succesions, trajs


def find_frames(traj, ts, states) -> tuple:
    frames = np.where([(s in states) for s in ts])[0]
    succesions = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
    succesions = [[suc[0], suc[-1]] for suc in succesions if len(suc) > traj.fps * 1]

    trajs = []
    for suc in succesions:
        traj_backroom = traj.cut_off([suc[0], suc[-1]])
        trajs.append(traj_backroom)
    return succesions, trajs


def plot_trj_color(ax, traj, c, cmap, norm, alpha=0.1):
    # map rainbow colors to the angle

    # all values in c larger than pi should be mapped to pi-value
    ax.scatter(traj.position[:, 0], traj.position[:, 1], c=c, cmap=cmap, norm=norm, s=1, alpha=alpha)
    # draw a black dot at starting point
    ax.scatter(traj.position[0, 0], traj.position[0, 1], c='black', s=5)
    ax.scatter(traj.position[-1, 0], traj.position[-1, 1], c='purple', s=5)


def open_fig(size, vmax=np.pi, cbar_label='angle'):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # vertical line at slits[size]
    ax.plot([slits[size][0], slits[size][0]], [0, arena_height[size] / 2 - exit_size[size] / 2],
            color='black', linestyle='--')
    ax.plot([slits[size][0], slits[size][0]], [arena_height[size], arena_height[size] / 2 + exit_size[size] / 2],
            color='black', linestyle='--')
    ax.plot([slits[size][1], slits[size][1]], [0, arena_height[size] / 2 - exit_size[size] / 2],
            color='black', linestyle='--')
    ax.plot([slits[size][1], slits[size][1]], [arena_height[size], arena_height[size] / 2 + exit_size[size] / 2],
            color='black', linestyle='--')
    ax.set_xlim([0, slits[size][0] * 1.6])
    ax.set_ylim([0, arena_height[size]])
    # set equal x and y scale
    ax.set_aspect('equal', adjustable='box')
    cmap = plt.get_cmap('rainbow')
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # draw colorbar
    cbar = plt.colorbar(sm)
    cbar.set_label(cbar_label)
    return fig, ax, cmap, norm


def plot_angle_in_back_room():
    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]
            fig, ax, cmap, norm = open_fig(size)

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                print(filename)
                traj = get(filename)
                s, trjs = find_backroom_frames(traj)
                for tr in trjs:
                    # fig, ax, cmap, norm = open_fig()
                    c = tr.angle % (2 * np.pi)
                    c[c > np.pi] = np.pi - (c[c > np.pi] - np.pi)
                    plot_trj_color(ax, tr, c, cmap, norm)
                    # plt.savefig('images\\angle_in_maze\\single\\' +
                    #             traj.filename + '_' + str(i) + '_' + str(tr.frames[0]) + '_' + str(tr.frames[-1]))
                    # plt.close()
            plt.savefig('images\\angle_in_maze\\' + 'reenter_' + str(size)[:1] + '.png')
            plt.close()
            # traj.play(wait=2)
            DEBUG = 1


def plot_angle_in_state_list(state_list):
    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]
            # fig, ax, cmap, norm = open_fig()

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                print(filename)
                traj = get(filename)
                s, trjs = find_frames(traj, time_series[traj.filename], state_list)

                fig, ax, cmap, norm = open_fig(size)
                for tr in trjs:
                    c = tr.angle % (2 * np.pi)
                    c[c > np.pi] = np.pi - (c[c > np.pi] - np.pi)
                    plot_trj_color(ax, tr, c, cmap, norm, alpha=0.01)
                plt.xlim([slits[size][0] * 0.8, slits[size][1] * 1.1])
                plt.ylim([arena_height[size]/2 - 3 * exit_size[size]/2, arena_height[size]/2 + 3 * exit_size[size]/2])
                plt.savefig('images\\angle_in_maze\\single\\' + state_list[0] + '\\' + traj.filename)
                plt.close()
            # plt.savefig('images\\angle_in_maze\\' + 'c_' + str(size)[:1] + '.png')
            # plt.close()
            # traj.play(wait=2)
            DEBUG = 1


def calculate_minimal_cumulative_distance(trajectory, distance_transform):
    # Compute the distance transform of the available area

    # Calculate cumulative distance of the trajectory to the boundary
    distance = []
    for point in tqdm(trajectory):
        ind1, ind2, ind3 = int(point[0]), int(point[1]), int(point[2])
        distance.append(distance_transform[ind1, ind2, ind3])
    return distance


def center_traj(traj_big, size):
    maze = Maze(traj_big)
    [shape_height, shape_width, shape_thickness, short_edge] = maze.getLoadDim()
    # where is the x.position[:, 0] at slits[size][0] +- maze.wallthick
    enter = np.where((traj_big.position[:, 0] > slits[size][0] - 1 * shape_thickness) &
                     (traj_big.position[:, 0] < slits[size][0] + 3 * shape_thickness))[0]
    # find the 0.02 and  98th quantiles of the y position
    mini, maxi = np.quantile(traj_big.position[enter, 1], [0.05, 0.95])
    # find distance of mean between mini and maxi to the center of the maze
    y_shift = -(np.mean([mini, maxi]) - arena_height[size] / 2)

    # y_shift = - (np.mean(traj_big.position[:, 1]) - arena_height[size]/2)
    traj_big.position[:, 1] = traj_big.position[:, 1] + y_shift

    h = centerOfMass_shift * shape_width
    # find the 90 % quantile of the x position
    maxi = np.quantile(traj_big.position[:, 0], 0.98)
    x_shift = (slits[size][1] - (maxi + 0.9 * (shape_width / 2 + h)))
    traj_big.position[:, 0] = traj_big.position[:, 0] + x_shift
    print(traj.filename, x_shift, y_shift)
    return traj_big, x_shift, y_shift


def plot_single_distance_graph(traj_big, size):
    fig, ax, cmap, norm = open_fig(size, vmax=1, cbar_label='distance [slit_distance/2]')
    plot_trj_color(ax, traj_big,
                   [di / (np.diff(slits[size]) / 2)[0] for di in d[traj_big.filename]],
                   cmap, norm, alpha=0.05)
    plt.xlim([slits[size][0] * 0.8, slits[size][1] * 1.1])
    plt.ylim([arena_height[size] / 2 - 3 * exit_size[size] / 2, arena_height[size] / 2 + 3 * exit_size[size] / 2])


if __name__ == '__main__':
    # plot_angle_in_back_room()
    # plot_angle_in_state_list(['c', 'cg'])
    d = {}
    x = {}
    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]
            # fig, ax, cmap, norm = open_fig()

            cs = ConfigSpace_Maze('ant', size.split(' ')[0], 'SPT',
                                  ('MazeDimensions_new2021_SPT_ant.xlsx',
                                   'LoadDimensions_new2021_SPT_ant.xlsx'))
            cs.load_space()
            distance_transform = distance_transform_edt(cs.space)
            ind_to_coor = cs.indices_to_coords(1, 0, 0)[0]

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                # filename = 'L_SPT_4660021_LSpecialT_1_ants (part 1)'
                print(filename)
                traj = get(filename)

                s, trjs = find_frames(traj, time_series[traj.filename], ['c', 'cg'])
                if len(trjs) != 0:
                    traj_big = trjs[0]
                    for tr in trjs[1:]:
                        traj_big = traj_big + tr

                    if len(traj_big.frames) / traj.fps > 60 * 3:  # longer than 5 min
                        traj_big, x_shift, y_shift = center_traj(traj_big, size)
                    else:
                        x_shift, y_shift = 0, 0

                    inds = [cs.coords_to_indices(x, y, theta) for x, y, theta in
                            zip(traj_big.position[:, 0], traj_big.position[:, 1], traj_big.angle)]

                    d[traj.filename] = calculate_minimal_cumulative_distance(inds, distance_transform)
                    d[traj.filename] = [di*ind_to_coor for di in d[traj.filename]]
                    x[traj.filename] = traj_big.position[:, 0].tolist()

                    plot_single_distance_graph(traj_big, size)
                    # x_shift only 2 digits after comma

                    plt.savefig('images\\distance_from_wall\\c\\' + traj_big.filename + '_' +
                                str(x_shift)[:4] + '_' + str(y_shift)[:4] + '.png', dpi=300)
                    plt.close()

            # plt.savefig('images\\angle_in_maze\\' + 'c_' + str(size)[:1] + '.png')
            # plt.close()
            # traj.play(wait=2)
            DEBUG = 1
    # save d
    with open('images\\distance_from_wall\\distance_c.json', 'w') as json_file:
        json.dump(d, json_file)
        json_file.close()
    with open('images\\distance_from_wall\\x_c.json', 'w') as json_file:
        json.dump(x, json_file)
        json_file.close()

