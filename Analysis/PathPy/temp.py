import pathpy as pp
from trajectory_inheritance.trajectory import get
from PhaseSpaces import PhaseSpace
from copy import copy
from pathpy.visualisation import plot, export_html

conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'SPT', name='')
conf_space.load_space()

erosion_radius = 9
conf_space_erode = copy(conf_space)
conf_space_erode.erode(radius=erosion_radius)

pss, centroids = conf_space_erode.split_connected_components()

conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space, conf_space_erode, pss, erosion_radius)
conf_space_labeled.load_space()

trajectories = [get(filename) for filename in ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]]

labels = [conf_space_labeled.label_trajectory(x) for x in trajectories]

paths = pp.Paths()
[paths.add_path(conf_space_labeled.reduces_labels(label)) for label in labels]

n = pp.Network.from_paths(paths)
export_html(n, 'paths.html')

hon_2 = pp.HigherOrderNetwork(paths, k=2, null_model=True)
print(hon_2)
export_html(hon_2, 'hon_2.html')

for e in hon_2.edges:
    print(e, hon_2.edges[e])

# t = pp.TemporalNetwork()
