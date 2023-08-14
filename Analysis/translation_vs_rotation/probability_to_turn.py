import numpy as np
from tqdm import tqdm
import pandas as pd
from Directories import home, network_dir
import json
from itertools import groupby
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import re
import os

centerOfMass_shift = - 0.08
SPT_ratio = 2.44 / 4.82

sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
                    'sim': ['S', 'M', 'L', 'XL'],
                    # 'sim': ['XS', 'S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium C', 'Large C', 'Medium NC', 'Large NC'],
                    }

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
# dict where states are integers
states_dict = {'ab': 1, 'ac': 1, 'b': 2, 'be': 2, 'b1': 2, 'b2': 2, 'c': 6, 'cg': 6, 'e': 8, 'eb': 8, 'eg': 8,
               'f': 11, 'g': 12, 'h': 13, 'i': 14}

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


def remove_subsequent_repetitions(lst):
    return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i - 1]]


def count_sequence_occurrences(numbers, order: str):
    sequence_count = 0

    joined_numbers = ''.join(map(str, numbers))
    substring = r'' + order
    # matches = re.findall(r'2[01]*2', joined_numbers)

    count = 0
    index = joined_numbers.find(substring)
    while index != -1:
        count += 1
        index = joined_numbers.find(substring, index + 1)
    return count

    #
    # for match in matches:
    #     sequence_count += 1 + match.count('2') + match.count('3')

    return len(matches)


def calc_b_to_b_or_c():
    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]
        b, c = {}, {}
        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                print(filename)
                filename = 'L_SPT_4660016_LSpecialT_1_ants (part 1)'
                ts = time_series_dict[filename]

                ts_n = [states_dict[s] for s in ts]
                ss_n = remove_subsequent_repetitions(ts_n)
                # number of times entered into b
                b[filename] = count_sequence_occurrences(ss_n, '212')
                # number of times entered into c
                c[filename] = count_sequence_occurrences(ss_n, '216')
                DEBUG = 1

        # save b and c
        with open('b_entrances.json', 'w') as json_file:
            json.dump(b, json_file)
            json_file.close()

        with open('c_entrances.json', 'w') as json_file:
            json.dump(c, json_file)
            json_file.close()

def calc_a_to_b_or_c():
    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]
        b, c = {}, {}
        for size in tqdm(sizes, desc='size'):
            df_size = df[df['size'] == size]

            for i, filename in enumerate(tqdm(df_size['filename'], desc=size)):
                print(filename)
                # filename = 'L_SPT_4660016_LSpecialT_1_ants (part 1)'
                ts = time_series_dict[filename]

                ts_n = [states_dict[s] for s in ts]
                ss_n = remove_subsequent_repetitions(ts_n)
                # number of times entered into b
                b[filename] = count_sequence_occurrences(ss_n, '12')
                # number of times entered into c
                c[filename] = count_sequence_occurrences(ss_n, '16')
                DEBUG = 1

        # save b and c
        with open('b_entrances_from_a.json', 'w') as json_file:
            json.dump(b, json_file)
            json_file.close()

        with open('c_entrances_from_a.json', 'w') as json_file:
            json.dump(c, json_file)
            json_file.close()


if __name__ == '__main__':
    # calc_b_to_a_or_c()
    # with open('b_entrances_from_a.json', 'r') as json_file:
    #     b = json.load(json_file)
    #     json_file.close()
    #
    # with open('c_entrances_from_a.json', 'r') as json_file:
    #     c = json.load(json_file)
    #     json_file.close()

    # calc_b_to_b_or_c()

    with open('b_entrances.json', 'r') as json_file:
        b = json.load(json_file)
        json_file.close()

    with open('c_entrances.json', 'r') as json_file:
        c = json.load(json_file)
        json_file.close()

    for solver_string in tqdm(['ant'], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]
        # exclude size 'Single (1)'
        df = df[df['size'] != 'Single (1)']
        df['b'] = df['filename'].apply(lambda x: b[x])
        df['c'] = df['filename'].apply(lambda x: c[x])

        # GROUP BY SIZE
        df_size_groups = df.groupby('size')

        # for every group add the 'b' and the 'c' column
        sums = df_size_groups.agg({'b': 'sum', 'c': 'sum'})

        # find c/(b+c) for every size
        sums['c/(b+c)'] = sums['c'] / (sums['b'] + sums['c'])
        sums['b/(b+c)'] = sums['b'] / (sums['b'] + sums['c'])

        # have 'XL' as the first row
        sums = sums.reindex(['XL', 'L', 'M', 'S (> 1)'])
        DEBUG = 2

