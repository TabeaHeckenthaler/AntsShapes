from Directories import home, network_dir
from matplotlib import pyplot as plt
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from itertools import groupby
from trajectory_inheritance.get import get
from colors import colors_humans as colors
from typing import Union
from Setup.Maze import Maze

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# date = '2023_06_27'
date = 'SimTrjs_RemoveAntsNearWall=False'
date1 = 'SimTrjs_RemoveAntsNearWall=True'
#
df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
df_gillespie1 = pd.read_excel(home + '\\Gillespie\\' + date1 + '_sim.xlsx')
df_human = pd.read_excel(home + '\\DataFrame\\final\\df_human.xlsx')
df_ant_excluded = pd.read_excel(home + '\\DataFrame\\final\\df_ant_excluded.xlsx')

color = {'AB': 'red', 'EG': 'grey', 'AC': 'green', 'F': 'blue'}
linestyle_solver = {'ant': '-', 'sim': 'dotted', 'human': '-'}
marker_solver = {'ant': '*', 'sim': '.', 'human': '*'}

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
                    'sim': ['S', 'M', 'L', 'XL'],
                    # 'sim': ['XS', 'S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium NC', 'Medium C', 'Large NC', 'Large C'],
                    }

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


def reduce_legend(ax):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def calc_state_series(time_series):
    return [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series])]


def extend_time_series_to_match_frames(ts, frame_len):
    indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                         for _ in range(frame_len)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def find_size(filename):
    return filename.split('_')[0]


def number_of_AB_mistakes(ss: list) -> Union[int, None]:
    """
    Quantify how often the shape undergoes the transition from ab -> b.
    :param ss: state series of the trajectory
    :return: number of times the shape went from ab to b.
    """
    # count the number of occurences of ['ab', 'b'] in the list ss
    if 'ab' not in ss:
        return None
    count = 0
    for i in range(len(ss) - 1):
        if ss[i] == 'ab' and ss[i + 1] == 'b':
            count += 1
    return count


def number_of_EG_mistakes(x, ts) -> Union[int, None]:
    """
    Quantify how often the shape entered to eg.
    :param ss: state series of trajectory
    :return: number of times the shape entered eg.
    """
    # count the number of occurences of ['e', 'eg'] in the list ss
    if 'e' not in ts:
        return None

    m = Maze(x)
    exit_size = m.exit_size
    height = m.arena_height
    corner1, corner2 = calc_corner_positions(x, m)

    # corner1[:, 1] enter [height/2 + exit_size/2, height/2 - exit_size/2]?
    c1 = (corner1[1, :] < height / 2 + exit_size / 2) & (corner1[1, :] > height / 2 - exit_size / 2)
    # corner2[:, 1] enter [height/2 + exit_size/2, height/2 - exit_size/2]?
    c2 = (corner2[1, :] < height / 2 + exit_size / 2) & (corner2[1, :] > height / 2 - exit_size / 2)

    # indices in e, eg states
    in_e_eg = np.logical_or(np.array(ts) == 'e', np.array(ts) == 'eg')

    in_eg = np.logical_and(in_e_eg, np.logical_or(c1, c2))
    in_eg_groups = group_boolean_list(in_eg)
    in_eg_groups = [i for i in in_eg_groups if i[1] - i[0] > x.fps]

    # x.play(frames=in_eg_groups[-3])

    # count = 0
    # for i in range(len(ss) - 1):
    #     if ss[i] == 'e' and ss[i + 1] == 'eg':
    #         count += 1
    return len(in_eg_groups)


def edge_locations(x):
    return [0, 0]


def calc_corner_positions(x, maze):
    # find frames, where only one edge_locations is behind the first slit
    [shape_height, shape_width, shape_thickness, short_edge] = maze.getLoadDim()

    centerOfMass_shift = - 0.08
    h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
    corners = np.array([[-shape_width / 2 - h, -shape_height / 2],
                        [-shape_width / 2 - h, shape_height / 2]])

    # find the position of the two corners of the shape in every frame
    corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
    corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])
    return corner1, corner2


def group_boolean_list(l):
    if not l.any():
        return []
    return [[i[0], i[-1]] for i in np.split(np.where(l)[0], np.where(np.diff(np.where(l)[0]) != 1)[0] + 1)]


def number_of_AC_mistakes(x, ts) -> Union[int, None]:
    """
    Quantify how often the shape attempted to enter from ac to c but did not rotate enough in order to enter.
    :param x: trajectory
    :return: number of times the shape attempted to enter from ac to c but did not rotate enough.
    """
    if 'c' not in ts:
        return None
    m = Maze(x)
    slit = m.slits[0]
    corner1, corner2 = calc_corner_positions(x, m)

    corner1_behind_slit = corner1[0, :] > slit
    corner2_behind_slit = corner2[0, :] > slit

    one_corner_behind_slit = np.logical_xor(corner1_behind_slit, corner2_behind_slit)
    both_corners_in_front_of_slit = np.logical_and(np.logical_not(corner1_behind_slit),
                                                   np.logical_not(corner2_behind_slit))
    # both_corners_behind_slit = np.logical_and(corner1_behind_slit, corner2_behind_slit)

    # group one_corner_behind_slit into groups of consecutive frames, listing the first and last frame of each group
    one_corner_behind_slit_groups = group_boolean_list(one_corner_behind_slit)

    # check whether before the first of after the last frame of each group,
    # both corners are behind the slit and remove those
    one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups \
                                     if len(both_corners_in_front_of_slit) > i[1] + 1 \
                                     and both_corners_in_front_of_slit[i[1] + 1] \
                                     and both_corners_in_front_of_slit[i[0] - 1]]

    # sort out those shorter than x.fps
    one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups if i[1] - i[0] > x.fps]
    # x.play(frames=one_corner_behind_slit_groups[-2])
    return len(one_corner_behind_slit_groups)


def number_of_F_mistakes(x, ts) -> Union[int, None]:
    """
    Quantify how often the shape attempted to enter from f to h but did not rotate enough in order to enter.
    :param x: trajectory
    :return: number of times the shape attempted to enter from f to h but did not rotate enough.
    """
    if 'h' not in ts:
        return None

    # find frames, where only one edge_locations is behind the first slit
    m = Maze(x)
    slit = m.slits[1]
    [shape_height, shape_width, shape_thickness, short_edge] = m.getLoadDim()

    centerOfMass_shift = - 0.08
    h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
    corners = np.array([[-shape_width / 2 - h, -shape_height / 2],
                        [-shape_width / 2 - h, shape_height / 2]])

    # find the position of the two corners of the shape in every frame
    corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
    corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])

    corner1_behind_slit = corner1[0, :] > slit
    corner2_behind_slit = corner2[0, :] > slit

    one_corner_behind_slit = np.logical_xor(corner1_behind_slit, corner2_behind_slit)
    one_corner_behind_slit_and_h = np.logical_and(one_corner_behind_slit, x.position[:, 0] > slit)

    # group one_corner_behind_slit into groups of consecutive frames, listing the first and last frame of each group
    one_corner_behind_slit_groups = [[i[0], i[-1]] for i in np.split(np.where(one_corner_behind_slit_and_h)[0],
                                     np.where(np.diff(np.where(one_corner_behind_slit_and_h)[0]) != 1)[0] + 1)]
    # sort out those shorter than x.fps
    one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups if i[1] - i[0] > x.fps]
    # x.play(frames=one_corner_behind_slit_groups[-5])
    return len(one_corner_behind_slit_groups)


def get_size_groups(df, solver) -> dict:
    if solver in ['ant', 'sim']:
        d = {size: group for size, group in df.groupby('size')}
        return d

    elif solver == 'human':
        # return dfs_human
        pass


def plot_number_of_mistakes(ax, AB, EG, AC, time, df, solver):
    dict_size_groups = get_size_groups(df, solver)
    means_AB, sem_AB = {}, {}
    means_EG, sem_EG = {}, {}
    means_AC, sem_AC = {}, {}
    means_CG, sem_CG = {}, {}

    # choose the 0.45 percentile of the fastest experiments
    for size, df_size in dict_size_groups.items():
        times = [time[filename] for filename in df_size['filename']]
        # find the 0.45 percentile of the fastest experiments
        time_45_percentile = np.percentile(times, 45)
        # find the filenames of the experiments that are faster than the 0.45 percentile
        fs_45_perc = [filename for filename in df_size['filename'] if time[filename] < time_45_percentile]
        means_AB[size] = np.mean([AB[filename] for filename in fs_45_perc if AB[filename] is not None])
        sem_AB[size] = np.std([AB[filename] for filename in fs_45_perc if AB[filename] is not None]) / \
                       np.sqrt(len([AB[filename] for filename in fs_45_perc if AB[filename] is not None]))
        means_EG[size] = np.mean([EG[filename] for filename in fs_45_perc if EG[filename] is not None])
        sem_EG[size] = np.std([EG[filename] for filename in fs_45_perc if EG[filename] is not None]) / \
                       np.sqrt(len([EG[filename] for filename in fs_45_perc if EG[filename] is not None]))
        means_AC[size] = np.mean([AC[filename] for filename in fs_45_perc if AC[filename] is not None])
        sem_AC[size] = np.std([AC[filename] for filename in fs_45_perc if AC[filename] is not None]) / \
                       np.sqrt(len([AC[filename] for filename in fs_45_perc if AC[filename] is not None]))
        means_CG[size] = np.mean([CG[filename] for filename in fs_45_perc if CG[filename] is not None])
        sem_CG[size] = np.std([CG[filename] for filename in fs_45_perc if CG[filename] is not None]) / \
                       np.sqrt(len([CG[filename] for filename in fs_45_perc if CG[filename] is not None]))

    if solver == 'ant':
        means_AB['S'], sem_AB['S'] = means_AB['S (> 1)'], sem_AB['S (> 1)']
        means_EG['S'], sem_EG['S'] = means_EG['S (> 1)'], sem_EG['S (> 1)']
        means_AC['S'], sem_AC['S'] = means_AC['S (> 1)'], sem_AC['S (> 1)']
        # means_F['S'], sem_F['S'] = means_F['S (> 1)'], sem_F['S (> 1)']

    ax.errorbar(sizes_per_solver['sim'], [means_AB[size] for size in sizes_per_solver[solver]],
                label='ab -> b: ' + solver, color=color['AB'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_AB[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['sim'], [means_EG[size] for size in sizes_per_solver[solver]],
                label='e -> eg: ' + solver, color=color['EG'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_EG[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['sim'], [means_AC[size] for size in sizes_per_solver[solver]],
                label='ac -> c: ' + solver, color=color['AC'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_AC[size] for size in sizes_per_solver[solver]])
    # ax.errorbar(sizes_per_solver['sim'], [means_F[size] for size in sizes_per_solver[solver]],
    #             label='F -> H: ' + solver, color=color['F'], marker=marker_solver[solver],
    #             linestyle=linestyle_solver[solver], yerr=[sem_F[size] for size in sizes_per_solver[solver]])


def calc():
    AB = {}
    AC = {}
    EG = {}
    F = {}
    time = {}

    for solver_string in tqdm(['ant', date1 + '_sim', ], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for filename in tqdm(df['filename']):
            print(filename)
            # filename = 'S_SPT_5190009_SSpecialT_1_ants (part 1)'
            ts = time_series_dict[filename]
            ss = calc_state_series(ts)

            AB[filename] = number_of_AB_mistakes(ss)
            traj = get(filename)
            time[filename] = traj.timer()
            ext_ts = extend_time_series_to_match_frames(ts, len(traj.frames))
            AC[filename] = number_of_AC_mistakes(traj, ext_ts)
            EG[filename] = number_of_EG_mistakes(traj, ext_ts)
            F[filename] = number_of_F_mistakes(traj, ext_ts)
            DEBUG = 1

    # save AB, AC, EG, F in json files
    with open('mistake_counting\\AB.json', 'w') as json_file:
        json.dump(AB, json_file)
        json_file.close()
    with open('mistake_counting\\AC.json', 'w') as json_file:
        json.dump(AC, json_file)
        json_file.close()

    with open('mistake_counting\\EG.json', 'w') as json_file:
        json.dump(EG, json_file)
        json_file.close()
    with open('mistake_counting\\F.json', 'w') as json_file:
        json.dump(F, json_file)
        json_file.close()
    with open('mistake_counting\\time.json', 'w') as json_file:
        json.dump(time, json_file)
        json_file.close()


if __name__ == '__main__':
    # calc()

    # load AB, AC, EG, F from json files
    with open('mistake_counting\\AB.json', 'r') as json_file:
        AB = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\AC.json', 'r') as json_file:
        AC = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\EG.json', 'r') as json_file:
        EG = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\CG.json', 'r') as json_file:
        CG = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\time.json', 'r') as json_file:
        time = json.load(json_file)
        json_file.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_number_of_mistakes(ax, AB, EG, AC, CG, time, dfs['ant'], 'ant')
    plot_number_of_mistakes(ax, AB, EG, AC, CG, time, dfs[date1 + '_sim'], 'sim')
    ax.set_xlabel('size')
    ax.set_ylabel('number of mistakes')
    ax.legend(prop={'size': 12})
    ax.set_ylim([0, 6])
    # reduce_legend(ax)
    plt.tight_layout()
    plt.savefig('mistake_counting\\number_of_mistakes.png', dpi=300)
    plt.savefig('mistake_counting\\number_of_mistakes.pdf', dpi=300)
    plt.savefig('mistake_counting\\number_of_mistakes.svg', dpi=300)
    DEBUG = 1