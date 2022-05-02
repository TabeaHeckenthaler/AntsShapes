from Directories import home, work_dir
import os
from os import path
import pickle


def get(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be unpickled
    :return: trajectory object
    """
    # this is local on your computer
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
