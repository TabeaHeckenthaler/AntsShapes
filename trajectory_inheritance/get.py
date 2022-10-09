from Directories import home, work_dir, SaverDirectories
import os
from os import path
import pickle
from trajectory_inheritance.trajectory_gillespie import TrajectoryGillespie
import numpy as np


def get(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be un-pickled
    :return: trajectory object
    """
    # this is local on your computer
    if filename.startswith('sim'):
        size, shape, solver = 'M', 'SPT', 'gillespie'
        with open(os.path.join(SaverDirectories['gillespie'], filename), 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        x = TrajectoryGillespie(size=size, shape=shape, filename=filename, fps=data.fps, winner=data.winner,
                                number_of_attached_ants=data.nAtt)

        if len(np.where(np.sum(data.position, axis=1) == 0)[0]):
            last_frame = np.where(np.sum(data.position, axis=1) == 0)[0][0]
        else:
            last_frame = -1
        x.position = data.position[:last_frame]
        x.angle = data.angle[:last_frame]
        x.frames = np.array([i for i in range(last_frame)])
        return x

    local_address = path.join(home, 'trajectory_inheritance', 'trajectories_local')
    if filename in os.listdir(local_address):
        with open(path.join(local_address, filename), 'rb') as f:
            print('You are loading ' + filename + ' from local copy.')
            x = pickle.load(f)
        return x

    # this is on labs network
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            if filename in os.listdir(path.join(root, dir)):
                address = path.join(root, dir, filename)
                with open(address, 'rb') as f:
                    x = pickle.load(f)
                return x
    else:
        raise ValueError('I cannot find ' + filename)
