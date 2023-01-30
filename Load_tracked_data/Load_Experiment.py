# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from matplotlib import pyplot as plt
from DataFrame.dataFrame import get_filenames
from os import listdir
from trajectory_inheritance.trajectory_human import Trajectory_human
from Directories import MatlabFolder, NewFileName, trackedHumanHandMovieDirectory
from trajectory_inheritance.trajectory_ant import Trajectory_ant
from trajectory_inheritance.trajectory_humanhand import Trajectory_humanhand
from trajectory_inheritance.get import get
from tqdm import tqdm
import json
from trajectory_inheritance.exp_types import exp_types
import os
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np
import pandas as pd


def is_extension(name) -> bool:
    if 'part' not in name:
        return False
    if int(name.split('part ')[1][0]) > 1 or 'r' in name.split('part ')[1]:
        return True


def get_image_analysis_results_files(solver, size, shape, free=False):
    if solver in ['ant', 'pheidole']:
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
    if solver in ['ant', 'pheidole', 'human', 'humanhand']:
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
    if solver in ['ant', 'pheidole']:
        x = Trajectory_ant(size=size, shape=shape, solver=solver, old_filename=filename, free=free, winner=winner,
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
        new = {name: bool(int(input(name + '   winner? '))) for name in unpickled if name not in winner_dict.keys()}

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
    if solver in ['ant', 'pheidole']:
        if 'ant' not in filename:
            iteration_number = str(int(filename.split('_')[-1][0]) + 1)
        else:
            iteration_number = str(int(filename.split('_')[-2][0]) + 1)
    else:
        iteration_number = str(int(filename.split('_')[-1][0]) + 1)
    part_number = str(int(filename.split('part ')[1][0]) + 1)

    same_movie = '_'.join(filename.split('_')[:2])
    next_movie = '_'.join(filename.split('_')[:1] + [movie_number])

    extension_candidates = [mat_file for mat_file in listdir(MatlabFolder(solver, size, shape, free=free))
                            if
                            (((same_movie in mat_file and iteration_number == mat_file.split('_')[-2][0])
                              or next_movie in mat_file)
                             and ('part ' + part_number in mat_file) and same_movie in mat_file)]
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


dir = os.getcwd() + '\\longest_jump.json'


def find_longest_jump(x) -> tuple:
    r = x.position
    distances = np.linalg.norm(np.diff(r, axis=0), axis=1)
    plt.close('all')
    plt.figure()
    plt.plot(distances)
    plt.pause(1)
    maxi_ind = int(np.argmax(distances))
    maxi = float(np.max(distances))
    return maxi, maxi_ind


def create_dict():
    dictio_p = {}
    ad = Altered_DataFrame()
    ad.choose_experiments(solver='ant')
    for filename in tqdm(ad.df['filename']):
        x = get(filename)
        dictio_p[filename] = find_longest_jump(x)

    with open(dir, 'w') as json_file:
        json.dump(dictio_p, json_file)
        json_file.close()


def find_most_problematic(d):
    df = pd.DataFrame(d).T
    df.columns = ['maxi', 'maxi_ind']
    df.sort_values(by='maxi', inplace=True, ascending=False)
    return df.head(n=26)


def correct_traj_many_mistakes(x):
    # CORRECT MANY MISTAKES
    with open(dir, 'r') as json_file:
        d = json.load(json_file)
        json_file.close()

    df_prob = find_most_problematic(d)
    for i in range(10, len(df_prob)):
        correct_traj(df_prob.iloc[i])

    # TO SMOOTH
    from scipy.signal import medfilt
    window = 9
    x.position[:, 0] = medfilt(x.position[:, 0], window)
    x.position[:, 1] = medfilt(x.position[:, 1], window)
    x.angle = medfilt(x.angle, window)

    filename = 'M_SPT_5050016_MSpecialT_1'
    x = get(filename)
    x.play(frames=[46016-300, 46016+300], wait=10)
    DEBUG = 1


def show_new_mistake(start, end):
    new_pos = x.position.copy()
    new_angle = x.angle.copy()
    plot_buffer = 200
    s, e = max(start - plot_buffer, 1), min(end + plot_buffer, len(new_angle))

    plt.figure()
    plt.plot(list(range(s, e)), new_pos[s: e, 0])
    plt.plot(list(range(s, e)), new_pos[s: e, 1])
    plt.plot(start, new_pos[start, 0], '*', markersize=10)
    plt.plot(end, new_pos[end, 0], '*', markersize=10)
    new_pos[start:end] = np.linspace(new_pos[start], new_pos[end], num=end - start)
    plt.plot(list(range(start, end)), new_pos[start: end, 0], color='k')
    plt.plot(list(range(start, end)), new_pos[start: end, 1], color='k')

    plt.figure()
    plt.plot(list(range(s, e)), new_angle[s: e], '.')
    plt.plot(start, new_angle[start], '*', markersize=10)
    plt.plot(end, new_angle[end], '*', markersize=10)
    new_angle[start:end] = np.linspace(new_angle[start], new_angle[end], num=end - start)
    plt.plot(list(range(start, end)), new_angle[start: end], '.', color='k')
    # plt.show()

    plt.pause(1)
    return new_pos, new_angle


def correct_traj(x):
    max_r, maxi_ind = find_longest_jump(x)
    while max_r > 0.2:
        print(max_r)
        print(x.VideoChain, x.frames[maxi_ind])
        buffer = 20
        start = max(0, maxi_ind - buffer)
        end = maxi_ind + buffer
        new_pos, new_angle = show_new_mistake(start, end)
        # new_pos, new_angle = show_new_mistake(np.where(x.frames == 1746)[0][0],
        #                                       np.where(x.frames == 5266)[0][0])
        # maxi_ind = int(x.fps * 0.25 * 5873)
        x.position = new_pos
        x.angle = new_angle
        max_r, maxi_ind = find_longest_jump(x)
    # x.play(frames=[start-50, end+50], wait=5)
    # x.position = x.position + np.array([-0.15, 0.1])
    # x.play(step=10)
    return x

# x.position[53184: 67948] = x.position[53184: 67948] + np.array([-0.2, 0])


with open('time_dictionary.txt', 'r') as json_file:
    time_dict = json.load(json_file)

with open('winner_dictionary.txt', 'r') as json_file:
    winner_dict = json.load(json_file)

x = get('S_SPT_5190011_SSpecialT_1_ants')
from scipy.signal import medfilt

window = 9
x.position[:, 0] = medfilt(x.position[:, 0], window)
x.position[:, 1] = medfilt(x.position[:, 1], window)
x.angle = medfilt(x.angle, window)

# x = get('S_SPT_4760017_SSpecialT_1_ants (part 1)')
# plt.plot(x.angle[43537: 54537])
# correct_traj(x)
# maxi_ind = int(x.fps * 0.25 * 5212)
# x.play(frames=[maxi_ind - 500, maxi_ind + 500, ], wait=10)
DEBUG = 1


if __name__ == '__main__':

    solver, shape, free = 'ant', 'SPT', False
    fps = {'human': 30, 'ant': 50, 'pheidole': 50, 'humanhand': 30}
    #
    # parts_ = ['LSPT_4650007_LSpecialT_1_ants (part 1).mat',
    #           'LSPT_4650007_LSpecialT_1_ants (part 1r).mat',
    #           'LSPT_4650008_LSpecialT_1_ants (part 2).mat',
    #           ]
    #
    # chain = [load(filename, 'ant', 'L', shape, fps[solver], [], winner=True) for filename in parts_]
    # x = chain[0]
    # for part in chain[1:]:
    #     x = x + part
    #
    # plt.plot(x.frames)
    # plt.show()
    # # x = x.add_missing_frames(chain, free)
    # x.play()
    # x.play(frames=[-250, -200], wait=10)
    # # x.angle = (x.angle + np.pi) % (2 * np.pi)
    # x.save()

    still_to_do = ['small_20220606162431_20220606162742_20220606162907_20220606163114.mat',]

    # for size in exp_types[shape][solver]:
    for size in ['L']:

        unpickled = find_unpickled(solver, size, shape)
        unpickled = ['LSPT_4660001_LSpecialT_1_ants (part 1).mat']
        winner_dict = continue_winner_dict(solver, shape)
        if len(unpickled) > 0:
            for results_filename in tqdm([u for u in unpickled if u not in still_to_do]):
                print(results_filename)
                parts_ = parts(results_filename, solver, size, shape)
                parts_ = ['LSPT_4660001_LSpecialT_1_ants (part 1).mat',
                          'LSPT_4660001_LSpecialT_1_ants (part 1r).mat',
                          'LSPT_4660002_LSpecialT_1_ants (part 2).mat']
                winner = winner_dict[results_filename]
                chain = [load(filename, solver, size, shape, fps[solver], [], winner=winner) for filename in parts_]
                x = chain[0]
                # x.position = x.position + np.array([-0.2, -0.1])
                x.play(wait=5)

                check = 1
                for part in chain[1:]:
                    print(part.filename)
                    # part.position = part.position + np.array([-0.1, 0])
                    part.play(wait=2)
                    # part.position = part.position + np.array([1.05, 0])
                    x = x + part

                plt.plot(x.frames, marker='.')
                plt.pause(1)

                # sometimes tracked only every 5th frame
                counts = np.bincount(np.abs(np.diff(x.frames)))
                x.fps = int(x.fps / np.argmax(counts))

                # x = x.add_missing_frames(chain, free)
                # x = x.cut_off([0, 4660])
                print(x.winner)

                # x.play(wait=2, step=5)
                # x.angle = (x.angle + np.pi) % (2 * np.pi)
                # x.angle[2717:3175] = (x.angle[2717:3175] + np.pi) % (2 * np.pi)
                # x.position[2717:3175, 0] = x.position[2717:3175, 0] - 0.1
                # plt.plot(x.frames)

                x = correct_traj(x)
                x.save()

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

