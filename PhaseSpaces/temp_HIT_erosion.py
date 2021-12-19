from trajectory_inheritance.trajectory import get
from PhaseSpaces import PhaseSpace
from copy import copy


conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'H', name='')
conf_space.load_space()
conf_space.visualize_space(colormap='Reds')

erosion_radius = 9
conf_space_erode = copy(conf_space)
conf_space_erode.erode(radius=erosion_radius)
conf_space_erode.visualize_space()

pss, centroids = conf_space_erode.split_connected_components()

conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space, conf_space_erode, pss, erosion_radius)
conf_space_labeled.load_space()
