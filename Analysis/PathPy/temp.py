import igraph
import numpy as np
from IPython.display import *
from IPython.display import HTML

import pathpy as pp
from trajectory_inheritance.trajectory import get
from PhaseSpaces import PhaseSpace
from copy import copy


conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'SPT', name='')
conf_space.load_space()

erosion_radius = 9
conf_space_erode = copy(conf_space)
conf_space_erode.erode(radius=erosion_radius)

pss, centroids = conf_space_erode.split_connected_components()

conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space, conf_space_erode, pss, erosion_radius)
conf_space_labeled.load_space()

x = get('XL_SPT_dil9_sensing4')
y = get('XL_SPT_dil9_sensing5')

labels_x = conf_space_labeled.label_trajectory(x)
labels_y = conf_space_labeled.label_trajectory(y)

# from itertools import zip_longest
# alternative = [i for i, j in zip_longest(labels_y[:-1], labels_y[1:]) if i[0] != j[0] or i[1] != j[1]]


#
# def stringify(labels) -> str:
#     st = ''
#     for label in labels:
#         st += str(label) + ','
#     return st


paths = pp.Paths()
paths.add_path(conf_space_labeled.reduces_labels(labels_x))
paths.add_path(conf_space_labeled.reduces_labels(labels_y))

paths.paths[0]  # gives the zero' th path and the number occurance of the
paths.paths[1]  # gives the zero' th path and the number occurance of the

n = pp.Network.from_paths(paths)

# t = pp.TemporalNetwork()
