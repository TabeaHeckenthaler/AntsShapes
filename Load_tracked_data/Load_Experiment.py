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
# from Load_tracked_data.PostTracking_Manipulations import SmoothConnector
import numpy as np
import json
from trajectory_inheritance.exp_types import exp_types
from trajectory_inheritance.trajectory_human import Trajectory_human
from trajectory_inheritance.trajectory_ant import Trajectory_ant
from PS_Search_Algorithms.D_star_lite import main


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

    mat_files = [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape))]
    new_names = [NewFileName(mat_file[:-4], solver, size, shape, expORsim) for mat_file in mat_files]

    unpickled = [mat_file
                 for new_name, mat_file in zip(new_names, mat_files)
                 if (new_name not in set(pickled) and not is_extension(new_name))]
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
        x = Trajectory_ant(size=size, shape=shape, old_filename=filename, free=free, winner=winner,
                           fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                           falseTracking=[falseTracking])
        if x.free:
            x.RunNum = int(input('What is the RunNumber?'))

    elif solver == 'human':
        shape = 'SPT'
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


def continue_winner_dict(solver, shape):
    with open('winner_dictionary.txt', 'r') as json_file:
        winner_dict = json.load(json_file)

    for size in exp_types[shape][solver]:
        print(size)
        unpickled = find_unpickled(solver, size, shape)
        new = {name: bool(input(name + '   winner? ')) for name in unpickled if name not in winner_dict.keys()}

        if len(new) > 0:
            winner_dict.update(new)
            with open('winner_dictionary.txt', 'w') as json_file:
                json.dump(winner_dict, json_file)
    return


def extension_exists(filename) -> list:
    """
    :return: list with candidates for being an extension.
    """
    movie_number = str(int(filename.split('_')[1]) + 1)
    part_number = str(int(filename.split('part ')[1][0]) + 1)

    same_movie = '_'.join(filename.split('_')[:2])
    next_movie = '_'.join(filename.split('_')[:1] + [movie_number])

    extension_candidates = [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape))
                            if
                            ((same_movie in mat_file or next_movie in mat_file) and 'part ' + part_number in mat_file)]
    if len(extension_candidates) > 1:
        input('to many extensions')
    return extension_candidates


if __name__ == '__main__':
    solver, shape = 'ant', 'SPT'
    if solver == 'human':
        fps = 30
    elif solver == 'ant':
        fps = 50
    else:
        fps = np.NaN

    for size in exp_types[shape][solver]:

        with open('winner_dictionary.txt', 'r') as json_file:
            winner_dict = json.load(json_file)

        for filename in tqdm(find_unpickled(solver, size, shape)):
            if filename not in winner_dict.keys():
                continue_winner_dict(solver, shape)

            winner = winner_dict[filename]

            print('\n' + filename)
            x = Load_Experiment(solver, filename, [], winner, fps, size=size, shape=shape)
            last_filename = filename

            if 'part ' in filename:
                while extension_exists(last_filename):
                    extension = extension_exists(last_filename)[0]
                    part1 = copy(x)
                    print('\n' + extension)
                    part2 = Load_Experiment(solver, extension, [], winner, fps, size=size, shape=shape)
                    connector_load = main(size=part1.size,
                                          shape=part1.shape,
                                          solver=part1.solver,
                                          sensing_radius=100,
                                          dil_radius=0,
                                          filename='shortest_path',
                                          starting_point=[part1.position[-1][0], part1.position[-1][1], part1.angle[-1]],
                                          ending_point=[part2.position[0][0], part2.position[0][1], part2.angle[0]],
                                          )
                    connector_load.stretch(17580)
                    connector_load.tracked_frames = connector_load.frames
                    connector_load.falseTracking = []
                    connector_load.free = part1.free
                    x = part1 + connector_load + part2

            x.play(step=20)

            # TODO: Check that the winner is correctly saved!!
            # TODO: add new file to contacts json file
            x.save()
