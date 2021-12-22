from PhaseSpaces import PhaseSpace
from copy import copy
import pathpy as pp


def load_labeled_conf_space():
    conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'SPT', name='')
    conf_space.load_space()

    erosion_radius = 9
    conf_space_erode = copy(conf_space)
    conf_space_erode.erode(radius=erosion_radius)

    pss, centroids = conf_space_erode.split_connected_components()

    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space, conf_space_erode, pss, erosion_radius)
    conf_space_labeled.load_space()
    return conf_space_labeled


def create_paths(labels) -> pp.paths:
    paths = pp.Paths()
    [paths.add_path(label) for label in labels]
    return paths


def create_higher_order_network(paths, k=2) -> pp.Network:
    hon = pp.HigherOrderNetwork(paths, k=k, null_model=True)

    for e in hon.edges:
        print(e, hon.edges[e])
    return hon
