from Analysis.PathPy.network_functions import *
import numpy as np
from Analysis.States import States


solver = 'ant'
size = 'XL'
shape = 'SPT'


def pathpy_analysis():
    # create paths and network
    paths = create_paths(labels)
    n = pp.Network.from_paths(paths)
    plot_network(n)
    state_order = ['b', 'a', 'd', 'e', 'f', 'g', 'j']
    state_dict = {v: n.node_to_name_map()[v] for v in state_order if v in n.node_to_name_map().keys()}

    # adjacency_matrix, degrees, laplacian, transition matrix and eigenvector
    A = n.adjacency_matrix().toarray()
    D = np.eye(len(n.degrees())) * np.array(n.degrees())
    L = n.laplacian_matrix(n.degrees()).toarray()
    T = n.transition_matrix().toarray()
    plot_transition_matrix(T, list(n.node_to_name_map().keys()) + ['i'])

    EV = n.leading_eigenvector(n.adjacency_matrix())

    # higher order networks
    hon = create_higher_order_network(paths)
    hon.likelihood(paths)
    eigenvalue_gap = pp.algorithms.spectral.eigenvalue_gap(hon)

    # How much slower is the second order network than the markovian network?
    pp.algorithms.path_measures.slow_down_factor(paths)

    # slow down factor
    # Ratios smaller than one indicate that the temporal network exhibits non - Markovian characteristics
    pp.algorithms.path_measures.entropy_growth_rate_ratio(paths, method='Miller')

    # Is the process Markovian?
    # estimate the order of the sequence (something like memory)
    ms = pp.MarkovSequence(paths.sequence())
    order = ms.estimate_order(4)


if __name__ == '__main__':
    # load labeled configuration space
    conf_space_labeled = load_labeled_conf_space(solver=solver, size=size, shape=shape, reduction=5)

    # load relevant experimental paths
    trajectories = get_trajectories(solver=solver, size=size, shape=shape, number=20)

    # label the trajectory according to conf_space_labeled
    labels = [States(conf_space_labeled, x, step=5 * x.fps) for x in trajectories]

    # TODO: add a time stamp to labels
    pathpy_analysis()

    # TODO: Look at returning to dead ends to define

