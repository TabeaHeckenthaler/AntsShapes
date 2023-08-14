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
from DataFrame.import_excel_dfs import dfs_human

plt.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman'})

# date = '2023_06_27'
date = 'SimTrjs_RemoveAntsNearWall=False'
date1 = 'SimTrjs_RemoveAntsNearWall=True'
#

centerOfMass_shift = - 0.08
SPT_ratio = 2.44 / 4.82

color = {'AB': 'red', 'EG': 'grey', 'B_STATES': 'purple', 'CG': 'green', 'F': 'blue'}
linestyle_solver = {'ant': '-', 'sim': 'dotted', 'human': '-'}
marker_solver = {'ant': '*', 'sim': '.', 'human': '*'}

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
                    'sim': ['S', 'M', 'L', 'XL'],
                    # 'sim': ['XS', 'S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium C', 'Large C', 'Medium NC', 'Large NC'],
                    }

with open(home + '\\ConfigSpace\\time_series_human.json', 'r') as json_file:
    time_series_human = json.load(json_file)
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


def edge_locations(x):
    return [0, 0]


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


def group_boolean_list(l):
    if not l.any():
        return []
    return [[i[0], i[-1]] for i in np.split(np.where(l)[0], np.where(np.diff(np.where(l)[0]) != 1)[0] + 1)]

#
# def number_of_AC_mistakes(x, ts) -> Union[int, None]:
#     """
#     Quantify how often the shape attempted to enter from ac to c but did not rotate enough in order to enter.
#     :param x: trajectory
#     :return: number of times the shape attempted to enter from ac to c but did not rotate enough.
#     """
#     if 'c' not in ts:
#         return None
#     m = Maze(x)
#     slit = m.slits[0]
#     corner1, corner2 = calc_corner_positions(x, m)
#
#     corner1_behind_slit = corner1[0, :] > slit
#     corner2_behind_slit = corner2[0, :] > slit
#
#     one_corner_behind_slit = np.logical_xor(corner1_behind_slit, corner2_behind_slit)
#     both_corners_in_front_of_slit = np.logical_and(np.logical_not(corner1_behind_slit),
#                                                    np.logical_not(corner2_behind_slit))
#     # both_corners_behind_slit = np.logical_and(corner1_behind_slit, corner2_behind_slit)
#
#     # group one_corner_behind_slit into groups of consecutive frames, listing the first and last frame of each group
#     one_corner_behind_slit_groups = group_boolean_list(one_corner_behind_slit)
#
#     # check whether before the first of after the last frame of each group,
#     # both corners are behind the slit and remove those
#     one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups \
#                                      if len(both_corners_in_front_of_slit) > i[1] + 1 \
#                                      and both_corners_in_front_of_slit[i[1] + 1] \
#                                      and both_corners_in_front_of_slit[i[0] - 1]]
#
#     # sort out those shorter than x.fps
#     one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups if i[1] - i[0] > x.fps]
#     # x.play(frames=one_corner_behind_slit_groups[-2])
#     return len(one_corner_behind_slit_groups)
#
#
# def number_of_F_mistakes(x, ts) -> Union[int, None]:
#     """
#     Quantify how often the shape attempted to enter from f to h but did not rotate enough in order to enter.
#     :param x: trajectory
#     :return: number of times the shape attempted to enter from f to h but did not rotate enough.
#     """
#     if 'h' not in ts:
#         return None
#
#     # find frames, where only one edge_locations is behind the first slit
#     m = Maze(x)
#     slit = m.slits[1]
#     [shape_height, shape_width, shape_thickness, short_edge] = m.getLoadDim()
#
#     centerOfMass_shift = - 0.08
#     h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
#     corners = np.array([[-shape_width / 2 - h, -shape_height / 2],
#                         [-shape_width / 2 - h, shape_height / 2]])
#
#     # find the position of the two corners of the shape in every frame
#     corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
#                         x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
#     corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
#                         x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])
#
#     corner1_behind_slit = corner1[0, :] > slit
#     corner2_behind_slit = corner2[0, :] > slit
#
#     one_corner_behind_slit = np.logical_xor(corner1_behind_slit, corner2_behind_slit)
#     one_corner_behind_slit_and_h = np.logical_and(one_corner_behind_slit, x.position[:, 0] > slit)
#
#     # group one_corner_behind_slit into groups of consecutive frames, listing the first and last frame of each group
#     one_corner_behind_slit_groups = [[i[0], i[-1]] for i in np.split(np.where(one_corner_behind_slit_and_h)[0],
#                                                                      np.where(np.diff(
#                                                                          np.where(one_corner_behind_slit_and_h)[
#                                                                              0]) != 1)[0] + 1)]
#     # sort out those shorter than x.fps
#     one_corner_behind_slit_groups = [i for i in one_corner_behind_slit_groups if i[1] - i[0] > x.fps]
#     return len(one_corner_behind_slit_groups)
#     # x.play(frames=one_corner_behind_slit_groups[-5])


def number_of_B_mistakes(traj, ext_ts):
    """
    Quantify how often the shape undergoes the transition from ab -> b.
    :param ss: state series of the trajectory
    :return: number of times the shape went from ab to b.
    """
    # count the number of occurences of ['ab', 'b'] in the list ss
    if 'ab' not in ext_ts:
        return None
    count = 0
    for i in range(len(ext_ts) - 1):
        if ext_ts[i] == 'ab' and ext_ts[i + 1] == 'b':
            count += 1
    return count


def number_of_B_STATES_mistakes(traj, ext_ts):
    """
    Quantify how often the shape checks a b_substate.
    :param traj:
    :param ext_ts:
    :return:
    """
    # m = Maze(traj)
    # slit = m.slits[0]
    # corner1, corner2 = calc_back_corner_positions(traj, m)

    # in_b1 = None  # corner 1 exited the [height/2 + exit_size/2, height/2 - exit_size/2] and larger than m.slits[-1]
    # in_b2 = None  # corner 2 exited the [height/2 + exit_size/2, height/2 - exit_size/2] and larger than m.slits[-1]
    # in_be = None  # corner 1 or 2 exited the [height/2 + exit_size/2, height/2 - exit_size/2] and smaller than m.slits[-1]

    # ts = np.array(['b' for _ in range(len(ext_ts))])
    # ts[in_b1] = 'b1'
    # ts[in_b2] = 'b2'
    # ts[in_be] = 'be'

    # # join to state_series
    # [''.join(ii[0]) for ii in groupby([tuple(label) for label in ts])]

    ss = [''.join(ii[0]) for ii in groupby([tuple(label) for label in ext_ts])]
    # count the number of occurrences of ['b1', 'b2', 'be'] in the list ss
    return len([s for s in ss if s in ['b1', 'b2', 'be']])


def number_of_AC_mistakes(traj, ext_ts):
    """
    Quantify whether the shape overturns in state ac
    :param traj:
    :param ext_ts:
    :return:
    """

    return 0


def number_of_CG_mistakes(traj, ext_ts):
    """
    Quantify whether the shape overturns in state ac
    :param traj:
    :param ext_ts:
    :return:
    """
    ss = [''.join(ii[0]) for ii in groupby([tuple(label) for label in ext_ts])]
    # count the number of occurrences of ['c', 'cg'] in the list ss
    count = 0
    for i in range(len(ss) - 1):
        if ss[i] == 'c' and ss[i + 1] == 'cg':
            count += 1
    return count


def number_of_EG_mistakes(traj, ext_ts) -> Union[int, None]:
    """
    Quantify how often the shape entered to eg.
    :param ss: state series of trajectory
    :return: number of times the shape entered eg.
    """
    # count the number of occurences of ['e', 'eg'] in the list ss
    if 'e' not in ext_ts:
        return None

    m = Maze(traj)
    exit_size = m.exit_size
    height = m.arena_height
    corner1, corner2 = calc_back_corner_positions(traj, m)

    # corner1[:, 1] enter [height/2 + exit_size/2, height/2 - exit_size/2]?
    c1 = (corner1[1, :] < height / 2 + exit_size / 2) & (corner1[1, :] > height / 2 - exit_size / 2)
    # corner2[:, 1] enter [height/2 + exit_size/2, height/2 - exit_size/2]?
    c2 = (corner2[1, :] < height / 2 + exit_size / 2) & (corner2[1, :] > height / 2 - exit_size / 2)

    # indices in e, eg states
    in_e_eg = np.logical_or(np.array(ext_ts) == 'e', np.array(ext_ts) == 'eg')
    in_eg = np.logical_and(in_e_eg, np.logical_or(c1, c2))
    in_eg_groups = group_boolean_list(in_eg)
    in_eg_groups = [i for i in in_eg_groups if i[1] - i[0] > traj.fps]

    # x.play(frames=in_eg_groups[-3])

    # count = 0
    # for i in range(len(ss) - 1):
    #     if ss[i] == 'e' and ss[i + 1] == 'eg':
    #         count += 1
    return len(in_eg_groups)


def number_of_F_H_mistakes(traj, ext_ts):
    """
    Quantify how often the shape returned to state e after entering f.
    :param traj:
    :param ext_ts:
    :return:
    """
    # count the number of occurences of ['ab', 'b'] in the list ss
    if 'f' not in ext_ts:
        return None
    count = 0
    for i in range(len(ext_ts) - 1):
        if ext_ts[i] == 'f' and ext_ts[i + 1] == 'e' and len(ext_ts) > i + 10 * traj.fps + 1 and \
                ext_ts[i + 10 * traj.fps] not in ['f', 'h']:
            count += 1
    return count


def get_size_groups(df, solver) -> dict:
    if solver in ['ant', 'sim']:
        d = {size: group for size, group in df.groupby('size')}
        return d

    elif solver == 'human':
        # return dfs_human
        pass


def plot_number_of_mistakes(ax, B, B_STATES, AC, EG, F_H, time, df, solver):
    means_B, sem_B = {}, {}
    means_B_STATES, sem_B_STATES = {}, {}
    means_CG, sem_CG = {}, {}
    means_EG, sem_EG = {}, {}
    means_F_H, sem_F_H = {}, {}
    # means_F, sem_F = {}, {}

    # choose the 0.45 percentile of the fastest experiments
    for size, df_size in dfs_human.items():
        # times = [time[filename] for filename in df_size['filename']]
        # find the filenames of the experiments that are faster than the 0.45 percentile
        fs = [filename for filename in df_size['filename']]
        means_B[size] = np.mean([B[filename] for filename in fs if B[filename] is not None])
        sem_B[size] = np.std([B[filename] for filename in fs if B[filename] is not None]) / \
                      np.sqrt(len([B[filename] for filename in fs if B[filename] is not None]))
        means_B_STATES[size] = np.mean([B_STATES[filename] for filename in fs if B_STATES[filename] is not None])
        sem_B_STATES[size] = np.std([B_STATES[filename] for filename in fs if B_STATES[filename] is not None]) / \
                             np.sqrt(len([B_STATES[filename] for filename in fs if B_STATES[filename] is not None]))
        means_CG[size] = np.mean([CG[filename] for filename in fs if CG[filename] is not None])
        sem_CG[size] = np.std([CG[filename] for filename in fs if CG[filename] is not None]) / \
                       np.sqrt(len([CG[filename] for filename in fs if CG[filename] is not None]))
        means_EG[size] = np.mean([EG[filename] for filename in fs if EG[filename] is not None])
        sem_EG[size] = np.std([EG[filename] for filename in fs if EG[filename] is not None]) / \
                       np.sqrt(len([AC[filename] for filename in fs if AC[filename] is not None]))
        means_F_H[size] = np.mean([F_H[filename] for filename in fs if F_H[filename] is not None])
        sem_F_H[size] = np.std([F_H[filename] for filename in fs if F_H[filename] is not None]) / \
                        np.sqrt(len([F_H[filename] for filename in fs if F_H[filename] is not None]))

    # if solver == 'ant':
    #     means_B['S'], sem_B['S'] = means_B['S (> 1)'], sem_B['S (> 1)']

    ax.errorbar(sizes_per_solver['human'], [means_B[size] for size in sizes_per_solver[solver]],
                label='ab: ' + solver, color=color['AB'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_B[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['human'], [means_B_STATES[size] for size in sizes_per_solver[solver]],
                label='b_states: ' + solver, color=color['B_STATES'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_B_STATES[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['human'], [means_AC[size] for size in sizes_per_solver[solver]],
                label='ac: ' + solver, color=color['AC'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_AC[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['human'], [means_CG[size] for size in sizes_per_solver[solver]],
                label='cg: ' + solver, color=color['CG'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_CG[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['human'], [means_EG[size] for size in sizes_per_solver[solver]],
                label='eg: ' + solver, color=color['EG'], marker=marker_solver[solver], markersize=10,
                linestyle=linestyle_solver[solver], yerr=[sem_EG[size] for size in sizes_per_solver[solver]])
    ax.errorbar(sizes_per_solver['human'], [means_F_H[size] for size in sizes_per_solver[solver]],
                label='f_h_f: ' + solver, color=color['F'], marker=marker_solver[solver],
                linestyle=linestyle_solver[solver], yerr=[sem_F_H[size] for size in sizes_per_solver[solver]])


def plot_number_of_mistakes_single(ax, d, solver, color, label):
    means_F_H, sem_F_H = {}, {}

    # choose the 0.45 percentile of the fastest experiments
    for size, df_size in dfs_human.items():
        # times = [time[filename] for filename in df_size['filename']]
        # find the filenames of the experiments that are faster than the 0.45 percentile
        fs = [filename for filename in df_size['filename']]
        means_F_H[size] = np.mean([d[filename] for filename in fs if d[filename] is not None])
        sem_F_H[size] = np.std([d[filename] for filename in fs if d[filename] is not None]) / \
                        np.sqrt(len([d[filename] for filename in fs if d[filename] is not None]))

    values = [means_F_H[size] for size in sizes_per_solver[solver]]
    ax.errorbar(sizes_per_solver[solver], values,
                label=label, color=color, marker=marker_solver[solver],
                linestyle=linestyle_solver[solver], yerr=[sem_F_H[size] for size in sizes_per_solver[solver]])
    ax.legend(prop={'size': 18})
    ax.plot([2.5, 2.5], [0, max(values)], color='black', linestyle='--')


def calc():
    B = {}  # enter_b_at_beginning
    B_STATES = {}  # checking_different_b_state
    AC = {}  # rotating_half_circle
    CG = {}  # checking transition from c to g
    EG = {}  # entering_eg
    F_H = {}  # not_exiting_f, 'large_20210413110406_20210413111655'
    time = {}

    for solver_string in tqdm(['human', ], desc='solver'):
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        df = pd.concat(dfs_human)

        for filename in tqdm(df['filename']):
            print(filename)
            # filename = `S_SPT_5190009_SSpecialT_1_ants (part 1)`
            ts = time_series_human[filename]

            traj = get(filename)
            B[filename] = number_of_B_mistakes(traj, ts)
            B_STATES[filename] = number_of_B_STATES_mistakes(traj, ts)
            AC[filename] = number_of_AC_mistakes(traj, ts)
            CG[filename] = number_of_CG_mistakes(traj, ts)
            EG[filename] = number_of_EG_mistakes(traj, ts)
            F_H[filename] = number_of_F_H_mistakes(traj, ts)
            DEBUG = 1

    # save AB, AC, EG, F in json files
    with open('mistake_counting\\B.json', 'w') as json_file:
        json.dump(B, json_file)
        json_file.close()

    with open('mistake_counting\\B_STATES.json', 'w') as json_file:
        json.dump(B_STATES, json_file)
        json_file.close()

    with open('mistake_counting\\AC.json', 'w') as json_file:
        json.dump(AC, json_file)
        json_file.close()

    with open('mistake_counting\\CG.json', 'w') as json_file:
        json.dump(CG, json_file)
        json_file.close()

    with open('mistake_counting\\EG.json', 'w') as json_file:
        json.dump(EG, json_file)
        json_file.close()

    with open('mistake_counting\\F_H.json', 'w') as json_file:
        json.dump(F_H, json_file)
        json_file.close()

    with open('mistake_counting\\time.json', 'w') as json_file:
        json.dump(time, json_file)
        json_file.close()


if __name__ == '__main__':
    calc()

    # load AB, AC, EG, F from json files
    with open('mistake_counting\\B.json', 'r') as json_file:
        B = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\B_STATES.json', 'r') as json_file:
        B_STATES = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\CG.json', 'r') as json_file:
        CG= json.load(json_file)
        json_file.close()
    with open('mistake_counting\\EG.json', 'r') as json_file:
        EG = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\F_H.json', 'r') as json_file:
        F_H = json.load(json_file)
        json_file.close()
    with open('mistake_counting\\time.json', 'r') as json_file:
        time = json.load(json_file)
        json_file.close()

    # fig, ax = plt.subplots(figsize=(5, 5))
    # plot_number_of_mistakes(ax, B, B_STATES, AC, EG, F_H, time, dfs_human, 'human')
    fig, axs = plt.subplots(4, 1, figsize=(5, 10), sharex=True)
    plot_number_of_mistakes_single(axs[0], B, 'human', 'red', 'enterances to b')
    plot_number_of_mistakes_single(axs[1], B_STATES, 'human', 'black', 'states explored in b (b1, b2, be)')
    # plot_number_of_mistakes_single(axs[2], AC, 'human', 'blue', 'AC, not implemented yet')
    plot_number_of_mistakes_single(axs[2], CG, 'human', 'blue', 'tried cg')
    plot_number_of_mistakes_single(axs[2], EG, 'human', 'purple', 'entered to eg')
    plot_number_of_mistakes_single(axs[3], F_H, 'human', 'green', 'exited f but returned to e')
    axs[-1].set_xlabel('size')
    axs[-1].set_ylabel('number of events')
    # ax.set_ylim([0, 6])
    # reduce_legend(ax)
    plt.tight_layout()
    plt.savefig('mistake_counting\\number_of_mistakes_humans.png', dpi=300)
    plt.savefig('mistake_counting\\number_of_mistakes_humans.pdf', dpi=300)
    plt.savefig('mistake_counting\\number_of_mistakes_humans.svg', dpi=300)
    DEBUG = 1
