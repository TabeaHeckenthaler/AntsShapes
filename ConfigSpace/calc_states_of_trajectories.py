import numpy as np
from tqdm import tqdm
import pandas as pd
import json
from itertools import groupby
from trajectory_inheritance.get import get
from Directories import home
from Setup.Maze import Maze

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

states = np.array(['ab', 'ac', 'b1', 'b2', 'be1', 'be2', 'b', 'cg', 'c', 'e', 'eb', "eg", 'f', 'h'])

state_transitions = {'ab': ['ac', 'b'],
                     'ac': ['ab', 'c'],
                     'b1': ['b2', 'be1', 'b'],
                     'b2': ['b1', 'be2', 'b'],
                     'be1': ['b1', 'be2', 'b'],
                     'be2': ['b2', 'be1', 'b'],
                     'b': ['b1', 'b2', 'be1', 'be2', 'ab'],
                     'cg': ['c', 'e'],
                     'c': ['cg', 'ac', 'e'],
                     'e': ['eb', 'eg', 'c', 'f', 'cg'],
                     'eb': ['e'],
                     'eg': ['e'],
                     'f': ['h', 'e'],
                     'h': ['f', 'i'],
                     'i': ['h']
                     }

def calc_state_series(time_series):
    return [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series])]


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

    corners = np.array([[shape_width / 2 - h, -short_edge / 2],
                        [shape_width / 2 - h, short_edge / 2]])

    # find the position of the two corners of the shape in every frame
    corner1 = np.array([x.position[:, 0] + corners[0, 0] * np.cos(x.angle) - corners[0, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[0, 0] * np.sin(x.angle) + corners[0, 1] * np.cos(x.angle)])
    corner2 = np.array([x.position[:, 0] + corners[1, 0] * np.cos(x.angle) - corners[1, 1] * np.sin(x.angle),
                        x.position[:, 1] + corners[1, 0] * np.sin(x.angle) + corners[1, 1] * np.cos(x.angle)])
    return corner1, corner2


def and_(arg1, arg2):
    return np.logical_and(arg1, arg2)


def or_(arg1, arg2):
    return np.logical_or(arg1, arg2)


def in_range(ymin, ymax, y):
    return np.logical_and(y > ymin, y < ymax)


def extend(b_cond, c_cond, b_name, c_name, ts, num_frames) -> list:
    a_name = ts[-1]
    b = np.where(b_cond)[0]
    c = np.where(c_cond)[0]

    if len(b) > 0:
        b_min = b[0]
    else:
        b_min = np.inf
    if len(c) > 0:
        c_min = c[0]
    else:
        c_min = np.inf

    # choose the smaller one and append 'a' until then
    if b_min == np.inf and c_min == np.inf:
        ts.extend([a_name] * (num_frames - len(ts)))
    elif b_min < c_min:
        ts.extend([a_name] * b_min)
        ts.append(b_name)
    else:
        ts.extend([a_name] * c_min)
        ts.append(c_name)
    return ts


def get_states(traj):
    maze = Maze(traj)
    back_corners = calc_back_corner_positions(traj, maze)
    back_corners = np.stack(back_corners, axis=1)
    front_corners = calc_front_corner_positions(traj, maze)
    front_corners = np.stack(front_corners, axis=1)
    cm = traj.position
    angle = traj.angle % (2 * np.pi)

    ch1_ch2 = maze.slits[0] + maze.wallthick / 2
    ch2_ch3 = maze.slits[1] + maze.wallthick / 2

    ymin, ymax = maze.arena_height / 2 - maze.exit_size / 2, maze.arena_height / 2 + maze.exit_size / 2

    ts = []
    i = 0

    if and_(*back_corners[0, :, i] < ch1_ch2) & and_(*front_corners[0, :, i] < ch1_ch2):
        ts.append('a')
    else:
        print('Not the beginning of the trajectory: ' + traj.filename)
        ts.append('a')

    while len(ts) < len(traj.frames):
        if ts[-1] == 'a':
            b_cond = or_(*front_corners[0, :, len(ts):] > ch1_ch2) & or_(
                *in_range(ymin, ymax, front_corners[1, :, len(ts):]))  #
            c_cond = and_(*back_corners[0, :, len(ts):] > ch1_ch2) & or_(
                *in_range(ymin, ymax, back_corners[1, :, len(ts):]))
            ts = extend(b_cond, c_cond, 'b', 'c', ts, len(traj.frames))

        if ts[-1] == 'b':
            a_cond = and_(*front_corners[0, :, len(ts):] < ch1_ch2)
            none_cond = np.zeros_like(a_cond, dtype=bool)
            ts = extend(a_cond, none_cond, 'a', 'z', ts, len(traj.frames))

        if ts[-1] == 'c':
            a_cond = and_(*back_corners[0, :, len(ts):] < ch1_ch2) & and_(*front_corners[0, :, len(ts):] < ch1_ch2)
            e_cond = and_(*front_corners[0, :, len(ts):] > ch1_ch2)
            ts = extend(a_cond, e_cond, 'a', 'e', ts, len(traj.frames))

        if ts[-1] == 'e':
            c_cond = and_(*back_corners[0, :, len(ts):] < ch1_ch2)
            f_cond = and_(*front_corners[0, :, len(ts):] > ch2_ch3) & or_(
                *in_range(ymin, ymax, front_corners[1, :, len(ts):]))
            ts = extend(c_cond, f_cond, 'c', 'f', ts, len(traj.frames))

        if ts[-1] == 'f':
            e_cond = or_(*front_corners[0, :, len(ts):] < ch2_ch3) & or_(
                *in_range(ymin, ymax, front_corners[1, :, len(ts):]))
            h_cond = and_(*back_corners[0, :, len(ts):] > ch2_ch3)
            ts = extend(e_cond, h_cond, 'e', 'h', ts, len(traj.frames))

        if ts[-1] == 'h':
            f_cond = or_(*back_corners[0, :, len(ts):] < ch2_ch3) & or_(
                *in_range(ymin, ymax, back_corners[1, :, len(ts):]))
            none_cond = np.zeros_like(f_cond, dtype=bool)
            ts = extend(f_cond, none_cond, 'f', 'z', ts, len(traj.frames))

    assert len(ts) == len(traj.frames)

    if ts[-1] == 'h':
        ts[-1] = 'i'
    ts = divide_into_substates(ts, traj)
    return ts


def divide_into_substates(ts, traj) -> list:
    maze = Maze(traj)
    back_corners = calc_back_corner_positions(traj, maze)
    back_corners = np.stack(back_corners, axis=1)
    front_corners = calc_front_corner_positions(traj, maze)
    front_corners = np.stack(front_corners, axis=1)
    ts = np.array(ts)
    cm = traj.position
    angle = traj.angle % (2 * np.pi)
    ch1_ch2 = maze.slits[0] + maze.wallthick / 2
    ch2_ch3 = maze.slits[1] + maze.wallthick / 2

    ymin, ymax = maze.arena_height / 2 - maze.exit_size / 2, maze.arena_height / 2 + maze.exit_size / 2

    a_mask = np.array(ts) == 'a'
    ac_mask = and_(np.pi / 2 < angle, angle < 3 * np.pi / 2)
    ab_mask = or_(angle < np.pi / 2, angle > 3 * np.pi / 2)
    # set
    ts = np.where(and_(a_mask, ab_mask), 'ab', ts)
    ts = np.where(and_(a_mask, ac_mask), 'ac', ts)

    b_mask = np.array(ts) == 'b'
    be1 = and_(and_(cm[:, 1] < maze.arena_height / 2, cm[:, 0] > ch1_ch2), ~and_(*in_range(ymin, ymax, front_corners[1, :, :])))
    be2 = and_(and_(cm[:, 1] > maze.arena_height / 2, cm[:, 0] > ch1_ch2), ~and_(*in_range(ymin, ymax, front_corners[1, :, :])))
    ts = np.where(and_(b_mask, be1), 'be1', ts)
    ts = np.where(and_(b_mask, be2), 'be2', ts)

    b1 = and_(cm[:, 1] < maze.arena_height / 2, and_(*front_corners[0, :, :] > ch2_ch3))
    b2 = and_(cm[:, 1] > maze.arena_height / 2, and_(*front_corners[0, :, :] > ch2_ch3))
    ts = np.where(and_(b_mask, b1), 'b1', ts)
    ts = np.where(and_(b_mask, b2), 'b2', ts)

    c_mask = np.array(ts) == 'c'
    cg_mask = and_(*back_corners[0, :, :] > 0.8 * (ch2_ch3 - ch1_ch2) + ch1_ch2)
    ts = np.where(and_(c_mask, cg_mask), 'cg', ts)

    e_mask = np.array(ts) == 'e'
    eb_mask = or_(*in_range(ymin, ymax, back_corners[1, :, :])) & or_(*back_corners[0, :, :] < ch1_ch2)
    eg_mask = or_(*in_range(ymin, ymax, back_corners[1, :, :])) & or_(*back_corners[0, :, :] > ch2_ch3)
    ts = np.where(and_(e_mask, eb_mask), 'eb', ts)
    ts = np.where(and_(e_mask, eg_mask), 'eg', ts)
    return ts.tolist()


def calc():
    for solver_string in tqdm(['ant','human',
                               'SimTrjs_RemoveAntsNearWall=True_sim',
                               'SimTrjs_RemoveAntsNearWall=False_sim'],
                              desc='solver', ):
        ts_dict = {}
        solver = solver_string.split('_')[-1]
        # sizes = sizes_per_solver[solver]
        df = dfs[solver_string]

        for filename in tqdm(df['filename']):
            print(filename)
            traj = get(filename)
            ts = get_states(traj)
            ts = divide_into_substates(ts, traj)
            ts_dict[filename] = ts
            # traj.play(ts=ts[45170:46000], frames=[45170, 46000], wait=20)
            # traj.play(ts=ts)
            DEBUG = 1
        # save in json file
        with open(f'time_series_{solver_string}.json', 'w') as f:
            json.dump(ts_dict, f)


if __name__ == '__main__':
    # calc()

    # filename = 'S_SPT_5190009_SSpecialT_1_ants (part 1)'
    filename = 'large_20220916093357_20220916093455'
    traj = get(filename)
    ts = get_states(traj)
    traj.play(ts=ts)

    DEBUG = 10
    DEBUG = 1