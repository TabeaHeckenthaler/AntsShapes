# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from DataFrame.dataFrame import get_filenames
from os import listdir
from Directories import MatlabFolder
from trajectory_inheritance.trajectory import sizes
from Directories import NewFileName
from tqdm import tqdm


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
    mat_files = listdir(MatlabFolder(solver, size, shape))
    return [mat_file for mat_file in mat_files if NewFileName(mat_file, size, shape, expORsim) not in set(pickled)]

    # TODO: Look also in the loaded movies, that were glued to one another.


def Load_Experiment(solver: str, filename: str, falseTracking: list, winner: bool, fps: int,
                    x_error: float = 0.0, y_error: float = 0.0, angle_error: float = 0.0,
                    size: str = None, shape: str = None, free: bool = False,
                    **kwargs):
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


new_starting_conditions = ['48000', '47900']
import numpy as np

if __name__ == '__main__':
    solver, shape = 'ant', 'SPT'
    # for size in sizes[solver]:
    for size in 'S':
        for filename in tqdm(find_unpickled(solver, size, shape)):
            if '48000' in filename:
                x = Load_Experiment(solver, filename, [], True, 50, size=size, shape=shape)
                x.play()
                save = 1
                # x.save()
