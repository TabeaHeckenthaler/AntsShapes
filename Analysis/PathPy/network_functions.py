from PhaseSpaces import PhaseSpace
from copy import copy
import pathpy as pp
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get


def load_labeled_conf_space(solver='ant', size='XL', shape='SPT', erosion_radius=9) -> PhaseSpace:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='')
    conf_space.load_space()

    conf_space_erode = copy(conf_space)
    conf_space_erode.erode(radius=erosion_radius)

    pss, centroids = conf_space_erode.split_connected_components()

    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space, pss, centroids, erosion_radius)
    conf_space_labeled.visualize_states()
    conf_space_labeled.load_space(uneroded_space=conf_space.space)

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


def get_trajectories(solver='human', size='Large', shape='SPT', number: int = 10) -> list:
    """
    Get trajectories based on the trajectories saved in myDataFrame.
    :param number: How many trajectories do you want to have in your list?
    :param solver: str
    :param size: str
    :param shape: str
    :return: list of objects, that are of the class or subclass Trajectory
    """
    df = myDataFrame[
        (myDataFrame['size'] == size) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver)]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]
