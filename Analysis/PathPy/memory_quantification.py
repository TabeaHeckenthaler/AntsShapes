from Analysis.PathPy.Network import *
from Analysis.PathPy.SPT_states import states, forbidden_transition_attempts, allowed_transition_attempts
import json
import pathpy as pp

# import pandas as pd
# from scipy.stats import chi2


transition_matrices = {}
# TODO: walk through the code, and try to understand what is happening.

solver, shape = 'human', 'SPT'

for size in ['Small Far', 'Medium', 'Large']:
    paths = PathWithoutSelfLoops(solver, size, shape, geometry)
    paths.load_paths()

    "all the possible states/nodes where proximity regions are  considered as states"
    state_order = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)

    "define network"
    my_network = Network.init_from_paths(state_order, solver, size, shape)

    "adds paths from the json to the network path statistics, assigns weight to the edges based on the number of " \
    "transition occurancies across all the paths"
    my_network.add_paths(paths)

    """create higher order model. 
    1. T: where each node represents a k length subpath
    2. T_tag: based on the 1st order model which in general allows different transitions
    """
    hon = my_network.create_higher_order_network(k=2, null_model=False)
    hon_null = my_network.create_higher_order_network(k=2, null_model=True)

    "Transition matrices"
    T = hon.transition_matrix().toarray()
    T_tag = hon_null.transition_matrix().toarray()

    "second largest eigenvalues"
    ls = np.linalg.eigvals(T)
    l_tags = np.linalg.eigvals(T_tag)

    l = np.sort(np.abs(ls))[-2]
    l_tag = np.sort(np.abs(l_tags))[-2]

    "diffusion speed up/slow down. If>1 slow down. if <1 speedup"
    S = np.log(l_tag) / np.log(l)
    print(S)

    """ 
        Actually https://www.pathpy.net/tutorial/model_selection.html describes how to select the best higher order model.
        Its not via the diffusion speedup/slow down but via statistical likhood. They already have the estimate of the order.
        Seems that for both cases best order that describes the data is second order model.
    """

    p = my_network.paths
    mog = pp.MultiOrderModel(p, max_order=10)
    print('Optimal order = ', mog.estimate_order())
