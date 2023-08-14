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

centerOfMass_shift = - 0.08
SPT_ratio = 2.44 / 4.82

sizes_per_solver = {'ant': ['XL', 'L', 'M', 'S (> 1)', ],
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


def plot_angle(ax, traj, cmap, norm):
    # map rainbow colors to the angle
    c = traj.angle % (2 * np.pi)

    # all values in c larger than pi should be mapped to pi-value
    c[c > np.pi] = np.pi - (c[c > np.pi] - np.pi)

    ax.scatter(traj.position[:, 0], traj.position[:, 1], c=c, cmap=cmap, norm=norm, s=1, alpha=0.1)
    # draw a black dot at starting point
    ax.scatter(traj.position[0, 0], traj.position[0, 1], c='black', s=5)
    ax.scatter(traj.position[-1, 0], traj.position[-1, 1], c='purple', s=5)


def open_fig():
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
    norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # draw colorbar
    cbar = plt.colorbar(sm)
    cbar.set_label('angle')
    return fig, ax, cmap, norm


if __name__ == '__main__':

    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]
            fig, ax, cmap, norm = open_fig()

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                print(filename)
                traj = get(filename)
                s, trjs = find_backroom_frames(traj)
                for tr in trjs:
                    # fig, ax, cmap, norm = open_fig()
                    plot_angle(ax, tr, cmap, norm)
                    # plt.savefig('images\\angle_in_maze\\single\\' +
                    #             traj.filename + '_' + str(i) + '_' + str(tr.frames[0]) + '_' + str(tr.frames[-1]))
                    # plt.close()
            plt.savefig('images\\angle_in_maze\\' + 'reenter_' + str(size)[:1] + '.png')
            plt.close()
            # traj.play(wait=2)
            DEBUG = 1
