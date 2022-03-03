import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle


def plot_velocity():
    """
    Load and plot velocity from json file, that Tabea sent on the 2nd of March.
    """
    with open(size + "_velocity_dict.json", 'r') as f:
        velocity_dict = json.load(f)
    fig, ax = plt.subplots(2)
    ax[0].hist(np.linalg.norm(np.vstack([velocity_dict['v_x'], velocity_dict['v_y']]), axis=0), density=True, bins=100)
    ax[1].hist(velocity_dict['omega'], density=True, bins=100)
    plt.show()


def get(filename, local_address):
    """
   Allows the loading of saved trajectory objects.
   :param filename: Name of the trajectory that is supposed to be unpickled
   :param local_address: adresse on your machine where pickle is saved (not on network)
   :return: trajectory object
   """
    if filename in os.listdir(local_address):
        with open(os.path.join(local_address, filename), 'rb') as f:
            print('You are loading ' + filename + 'from local copy.')
            x = pickle.load(f)
        return x
    else:
        raise ValueError('Cant find trajectory in ', folder)


if __name__ == '__main__':
    size = 'XL'

    # free trajectories
    home = os.path.join(os.path.abspath(__file__).split('\\')[0] + os.path.sep,
                        *os.path.abspath(__file__).split(os.path.sep)[1:-3])

    # I suggest, that you put the trajectory pickles into the following directory
    folder = os.path.join('trajectory_inheritance', 'trajectories_local')

    name_of_exp = 'XL_SPT_4280003_freeXLSpecialT_1'
    x = get(name_of_exp, os.path.join(home, folder))
    x.play()

    # plot velocity from the histograms I sent
    plot_velocity()
