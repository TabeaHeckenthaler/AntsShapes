# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from DataFrame.dataFrame import get_filenames
from os import listdir
from Directories import MatlabFolder, NewFileName, trackedHumanHandMovieDirectory
from tqdm import tqdm
import json
from trajectory_inheritance.exp_types import exp_types
from trajectory_inheritance.trajectory_human import Trajectory_human
from trajectory_inheritance.trajectory_ant import Trajectory_ant
from trajectory_inheritance.trajectory_humanhand import Trajectory_humanhand


def is_extension(name) -> bool:
    if 'part' not in name:
        return False
    if int(name.split('part ')[1][0]) > 1:
        return True


def get_image_analysis_results_files(solver, size, shape, free=False):
    if solver == 'ant':
        return [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape, free=free)) if
                size + shape in mat_file]
    elif solver == 'human':
        human_size_dict = {'Large': 'large', 'Medium': 'medium', 'Small Near': 'small', 'Small Far': 'small2'}
        return [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape, free=free)) if
                human_size_dict[size] in mat_file]
    elif solver == 'humanhand':
        return listdir(trackedHumanHandMovieDirectory)
    else:
        raise Exception('Where are you mat files for the solver ' + solver)


def find_unpickled(solver, size, shape, free=False):
    """
    find the .mat files, which are not pickled yet.
    :return: list of un-pickled .mat file names (without .mat extension)
    """
    pickled = get_filenames(solver, size=size, free=free)
    if solver in ['ant', 'human', 'humanhand']:
        expORsim = 'exp'
    else:
        expORsim = 'sim'

    results_files = get_image_analysis_results_files(solver, size, shape, free=free)
    new_names = [NewFileName(results_files[:-4], solver, size, shape, expORsim) for results_files in results_files]

    unpickled = [results_file
                 for new_name, results_file in zip(new_names, results_files)
                 if (new_name not in set(pickled)
                     and not is_extension(new_name)
                     and not results_file.startswith('SSPT_45100'))]
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
        # if x.free:
        #     x.RunNum = int(input('What is the RunNumber?'))

    elif solver == 'human':
        shape = 'SPT'
        x = Trajectory_human(size=size, shape=shape, filename=filename, winner=winner,
                             fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                             falseTracking=falseTracking)

    elif solver == 'humanhand':
        shape = 'SPT'
        x = Trajectory_humanhand(size=size, shape=shape, filename=filename, winner=winner, fps=fps, x_error=x_error,
                                 y_error=y_error, angle_error=angle_error, falseTracking=falseTracking)
    else:
        raise Exception('Not a valid solver')

    x.matlab_loading(filename)  # this is already after we added all the errors...

    # if 'frames' in kwargs:
    #     frames = kwargs['frames']
    # else:
    #     frames = [x.frames[0], x.frames[-1]]
    # if len(frames) == 1:
    #     frames.append(frames[0] + 2)
    # f1, f2 = int(frames[0]) - int(x.frames[0]), int(frames[1]) - int(x.frames[0]) + 1
    # x.position, x.angle, x.frames = x.position[f1:f2, :], x.angle[f1:f2], x.frames[f1:f2]
    return x


def part2_filename(part1_filename):
    l = part1_filename.split("_")
    return ''.join(l[:1]) + '_' + str(int(l[1]) + 1) + '_' + '_'.join(l[2:-1]) + "_" + l[-1].replace('1', '2')


def continue_winner_dict(solver, shape) -> dict:
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
    return winner_dict


def extension_exists(filename, solver, size, shape, free=False) -> list:
    """
    :return: list with candidates for being an extension.
    """
    movie_number = str(int(filename.split('_')[1]) + 1)
    iteration_number = str(int(filename.split('_')[-1][0]) + 1)
    part_number = str(int(filename.split('part ')[1][0]) + 1)

    same_movie = '_'.join(filename.split('_')[:2])
    next_movie = '_'.join(filename.split('_')[:1] + [movie_number])

    extension_candidates = [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape, free=free))
                            if
                            (((same_movie in mat_file and iteration_number == mat_file.split('_')[-1][0])
                              or next_movie in mat_file)
                             and 'part ' + part_number in mat_file)]
    if len(extension_candidates) > 1:
        raise ValueError('to many extensions')
    return extension_candidates


def parts(filename, solver, size, shape, free=False):
    VideoChain = [filename]
    if 'part ' in filename:
        while extension_exists(VideoChain[-1], solver, size, shape, free=free):
            VideoChain.append(extension_exists(VideoChain[-1], solver, size, shape, free=free)[0])
    return VideoChain


def load(filename, solver, size, shape, fps, falseTracking, winner=None, free=False):
    if not free:
        if winner is None:
            winner_dict = continue_winner_dict(solver, shape)
            winner = winner_dict[filename]
    filename = filename[:-4]
    print('\n' + filename)
    return Load_Experiment(solver, filename, falseTracking, winner, fps, size=size, shape=shape, free=free)


with open('time_dictionary.txt', 'r') as json_file:
    time_dict = json.load(json_file)

with open('winner_dictionary.txt', 'r') as json_file:
    winner_dict = json.load(json_file)

if __name__ == '__main__':

    solver, shape = 'humanhand', 'SPT'
    fps = {'human': 30, 'ant': 50, 'humanhand': 30}

    for size in exp_types[shape][solver]:
        unpickled = find_unpickled(solver, size, shape)
        if len(unpickled) > 0:
            for results_filename in tqdm(unpickled):
                print(results_filename)
                x = load(results_filename, solver, size, shape, fps[solver], [])
                chain = [x] + [load(filename, solver, size, shape, fps[solver], [], winner=x.winner)
                               for filename in parts(results_filename, solver, size, shape)[1:]]
                x = x.add_missing_frames(chain)
                x.play(wait=5)
                x.save()

                # TODO: Check that the winner is correctly saved!!
                # TODO: add new file to contacts json file
                # TODO: add new file to pandas DataFrame

    # free trajectories
    # solver, shape = 'ant', 'SPT'
    # size = 'XL'
    # free = True
    # fps = 50
    #
    # for results_filename in tqdm(find_unpickled(solver, size, shape, free=True)):
    #     print(results_filename)
    #     x = load(results_filename, solver, size, shape, fps, falseTracking=[], free=free)
    #     chain = [x] + [load(filename, solver, size, shape, fps, [], winner=x.winner, free=free)
    #                    for filename in parts(results_filename, solver, size, shape, free=free)[1:]]
    #
    #     x = x.add_missing_frames(chain, free=free)
    #     x.play()
    #     x.save(address=path.join(SaverDirectories[solver][free], x.filename))

