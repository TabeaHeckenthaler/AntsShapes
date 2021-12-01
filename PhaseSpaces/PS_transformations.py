from PhaseSpaces import PhaseSpace
from copy import copy
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'SPT', name='eroded')
    conf_space.load_space()
    conf_space.visualize_space(colormap='Oranges')

    conf_space_erode = copy(conf_space)
    conf_space_erode.erode(radius=9)
    # fig = conf_space_erode.visualize_space(fig=conf_space.fig)

    # TODO: Take into account the periodicity
    pss, centroids = conf_space_erode.split_connected_components()

    # TODO: draw center of this thing :)
    for index in range(len(centroids)):
        pss[index].visualize_space(fig=conf_space.fig)
        conf_space.draw(centroids[index][:2], centroids[index][-1], scale_factor=1)
        k = 1

    # TODO: Are there network packages I can map this to?

    # TODO: A given trajectory into runs between nodes of network

    # TODO: Given all trajectories, what

    k = 1
