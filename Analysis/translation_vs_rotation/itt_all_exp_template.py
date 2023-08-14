import numpy as np
from tqdm import tqdm
import pandas as pd
from Directories import home, network_dir
import json
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import os

centerOfMass_shift = - 0.08
SPT_ratio = 2.44 / 4.82

sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
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


if __name__ == '__main__':

    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for filename in tqdm(df['filename']):
            print(filename)
            traj = get(filename)

            DEBUG = 1
