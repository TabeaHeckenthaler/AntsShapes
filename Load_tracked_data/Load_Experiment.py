# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from DataFrame.dataFrame import get_filenames
from os import listdir
from Directories import MatlabFolder
from Directories import NewFileName
from tqdm import tqdm
from copy import copy
from Load_tracked_data.PostTracking_Manipulations import SmoothConnector
import numpy as np


def is_extension(name) -> bool:
    if 'part' not in name:
        return False
    if int(name.split('part ')[1][0]) > 1:
        return True


def find_unpickled(solver, size, shape):
    """
    find the .mat files, which are not pickled yet.
    :return: list of un-pickled .mat file names (without .mat extension)
    """
    pickled = get_filenames(solver, size=size)
    if solver in ['ant', 'human']:
        expORsim = 'exp'
    else:
        expORsim = 'sim'

    mat_files = [mat_file[:-4] for mat_file in listdir(MatlabFolder(solver, size, shape))]
    new_names = [NewFileName(mat_file, solver, size, shape, expORsim) for mat_file in mat_files]

    unpickled = [new_name for new_name in new_names if (new_name not in set(pickled) and not is_extension(new_name))]
    return unpickled


def Load_Experiment(solver: str, filename: str, falseTracking: list, winner: bool, fps: int,
                    x_error: float = 0.0, y_error: float = 0.0, angle_error: float = 0.0,
                    size: str = None, shape: str = None, free: bool = False,
                    **kwargs):
    """
    solver: str 'ant', 'human', 'humanhand', 'gillespie'
    filename: old_filename with .mat extension
    falseTracking: list of lists [[fr1, fr2], [fr3, fr4] ...] with falsely tracked frames
    winner: boolean, whether the trail was successful
    fsp: frames per second of the camera
    """
    if solver == 'ant':
        from trajectory_inheritance.trajectory_ant import Trajectory_ant
        x = Trajectory_ant(size=size, shape=shape, old_filename=filename, free=free, winner=winner,
                           fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                           falseTracking=[falseTracking])
        if x.free:
            x.RunNum = int(input('What is the RunNumber?'))

    elif solver == 'human':
        shape = 'SPT'
        from trajectory_inheritance.trajectory_human import Trajectory_human
        x = Trajectory_human(size=size, shape=shape, filename=filename, winner=winner,
                             fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                             falseTracking=falseTracking)

    else:
        raise Exception('Not a valid solver')

    x.matlab_loading(filename)  # this is already after we added all the errors...

    if 'frames' in kwargs:
        frames = kwargs['frames']
    else:
        frames = [x.frames[0], x.frames[-1]]
    if len(frames) == 1:
        frames.append(frames[0] + 2)
    f1, f2 = int(frames[0]) - int(x.frames[0]), int(frames[1]) - int(x.frames[0]) + 1
    x.position, x.angle, x.frames = x.position[f1:f2, :], x.angle[f1:f2], x.frames[f1:f2]
    return x


def part2_filename(part1_filename):
    l = part1_filename.split("_")
    return ''.join(l[:1]) + '_' + str(int(l[1]) + 1) + '_' + '_'.join(l[2:-1]) + "_" + l[-1].replace('1', '2')


if __name__ == '__main__':
    solver, shape = 'ant', 'SPT'
    if solver == 'human':
        fps = 30
    elif solver == 'ant':
        fps = 50
    else:
        fps = np.NaN
    # for size in sizes[solver]:
    for size in ['XL']:
        for filename in tqdm(find_unpickled(solver, size, shape)):
            print('\n' + filename)
            winner = bool(input('winner? '))
            x = Load_Experiment(solver, filename, [], winner, fps, size=size, shape=shape)
            if 'force_vector 1' in filename:
                part1 = copy(x)
                print('\n' + part2_filename(filename))
                part2 = Load_Experiment(solver, part2_filename(filename), [], winner, fps, size=size, shape=shape)
                con = SmoothConnector(part1, part2, con_frames=1000)
                x = part1 + con + part2
            x.play(step=20)
            DEBUG = 1
            # TODO: Check that the winner is correctly saved!!
            # TODO: add new file to contacts json file
            x.save()
