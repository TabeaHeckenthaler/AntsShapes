from trajectory_inheritance.get import get
from tqdm import tqdm
import json
import os
from matplotlib import pyplot as plt
from DataFrame.Altered_DataFrame import Altered_DataFrame
import numpy as np
import pandas as pd

dir = os.getcwd() + '\\longest_jump.json'


def find_longest_jump(x) -> tuple:
    r = x.position
    distances = np.linalg.norm(np.diff(r, axis=0), axis=1)
    plt.close('all')
    plt.figure()
    plt.plot(distances)
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


def correct_traj(s):
    x = get(s.name)

    maxi_ind = int(s['maxi_ind'])
    # maxi_ind = 53422

    # x.frames[maxi_ind]
    # maxi_ind = 37529
    buffer = 10
    start = max(0, maxi_ind - buffer)
    end = maxi_ind + buffer
    # start = 4500
    # end = 2000

    new_pos = x.position.copy()
    new_angle = x.angle.copy()

    plot_buffer = 100
    plt.figure()
    s, e = max(start - plot_buffer, 1), end + plot_buffer
    plt.plot(list(range(s, e)), new_pos[s: e, 0])
    plt.plot(list(range(s, e)), new_pos[s: e, 1])
    plt.plot(start, new_pos[start, 0], '*', markersize=10)
    plt.plot(end, new_pos[end, 0], '*', markersize=10)

    plt.figure()
    plt.plot(list(range(s, e)), new_angle[s: e], '.')
    plt.plot(start, new_angle[start], '*', markersize=10)
    plt.plot(end, new_angle[end], '*', markersize=10)

    # plt.figure()
    # plt.plot(new_pos[s: e, 0], new_pos[s: e, 1])

    new_pos[start:end] = np.linspace(new_pos[start], new_pos[end], num=end-start)
    new_angle[start:end] = np.linspace(new_angle[start], new_angle[end], num=end-start)

    x.position = new_pos
    x.angle = new_angle
    x.save()
    x.play(frames=[s, e], wait=5)
    x.play(step=1)
    # mid = 24430
    # pos_24430 = [2.8, 3.1]
    # ang_24430 = 2.42920
    #
    # from Setup.Maze import Maze
    # m = Maze(x)
    # m.set_configuration(position=pos_24430, angle=ang_24430)
    # m.draw()
    #
    # new_pos[start:mid] = np.linspace(new_pos[start], pos_24430, num=mid - start)
    # new_angle = x.angle.copy()
    # new_angle[start:mid] = np.linspace(new_angle[start], ang_24430, num=mid - start)
    #
    # new_pos[mid - 1:end] = np.linspace(new_pos[mid - 1], new_pos[end], num=end - mid + 1)
    # new_angle[mid - 1:end] = np.linspace(new_angle[mid - 1], new_angle[end], num=end - mid + 1)

    DEBUG = 1


if __name__ == '__main__':
    # create_dict()
    filename = 'M_SPT_5050016_MSpecialT_1'
    x = get(filename)
    # x.play()
    max_r, maxi_ind = find_longest_jump(x)

    with open(dir, 'r') as json_file:
        d = json.load(json_file)
        json_file.close()

    df_prob = find_most_problematic(d)
    for i in range(10, len(df_prob)):
        correct_traj(df_prob.iloc[i])

    # TO SMOOTH
    from scipy.signal import medfilt
    x.position[:, 0] = medfilt(x.position[:, 0], 109)
    x.position[:, 0] = medfilt(x.position[:, 0], 109)
    x.angle = medfilt(x.angle, 109)

    filename = 'M_SPT_5050016_MSpecialT_1'
    x = get(filename)
    x.play(frames=[46016-300, 46016+300], wait=10)
    DEBUG = 1
