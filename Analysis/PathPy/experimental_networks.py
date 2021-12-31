from Analysis.PathPy.network_functions import *
from pathpy.visualisation import plot, export_html
from Directories import network_dir
import igraph
import numpy as np

solver = 'human'
size = 'Large'
shape = 'SPT'

if __name__ == '__main__':
    # load labeled configuration space
    conf_space_labeled = load_labeled_conf_space(solver=solver, size=size, shape=shape)

    # load relevant experimental paths
    trajectories = get_trajectories(solver=solver, size=size, shape=shape, number=20)

    # label the trajectory according to conf_space_labeled
    labels = [conf_space_labeled.reduces_labels(conf_space_labeled.label_trajectory(x)) for x in trajectories]

    # create paths and network
    paths = create_paths(labels)
    n = pp.Network.from_paths(paths)
    state_dict = n.node_to_name_map()

    # adjacency_matrix, degrees, laplacian, transition matrix and eigenvector
    A = n.adjacency_matrix().toarray()
    D = np.eye(len(n.degrees())) * np.array(n.degrees())
    L = n.laplacian_matrix(n.degrees()).toarray()
    T = n.transition_matrix().toarray()
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

    # TODO: Look at returning to dead ends to define

    # # display network
    # def Network2igraph(network):
    #     """
    #     Returns an igraph Graph object which represents
    #     the k-th layer of a multi-order graphical model.
    #     """
    #     g = igraph.Graph(directed=True)
    #
    #     for e in network.edges:
    #         if g.vcount() == 0 or e[0] not in g.vs()["name"]:
    #             g.add_vertex(e[0])
    #         if g.vcount() == 0 or e[1] not in g.vs()["name"]:
    #             g.add_vertex(e[1])
    #         g.add_edge(e[0], e[1], weight=network.edges[e]['weight'])
    #     return g
    #
    #
    # g1 = Network2igraph(n)
    # visual_style = {}
    # visual_style["layout"] = g1.layout_auto()
    # visual_style["vertex_label"] = g1.vs["name"]
    # visual_style["edge_label"] = g1.es["weight"]
    #
    # export_html(n, path.join('network.html'), **visual_style)
    # export_html(hon, path.join(network_dir, 'hon.html'), **visual_style)
