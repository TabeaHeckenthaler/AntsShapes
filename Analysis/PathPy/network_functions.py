import pathpy as pp
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get
from matplotlib import pyplot as plt
import igraph
from pathpy.visualisation import export_html
from Directories import network_dir
from os import path
from Analysis.PathPy.AbsorbingMarkovChain import *
from Analysis.GeneralFunctions import graph_dir
import os


def create_paths(labels) -> pp.paths:
    paths = pp.Paths()
    [paths.add_path(label) for label in labels]
    return paths


def create_higher_order_network(paths, k=2) -> pp.Network:
    hon = pp.HigherOrderNetwork(paths, k=k, null_model=True)

    # for e in hon.edges:
    #     print(e, hon.edges[e])
    return hon


def get_trajectories(solver='human', size='Large', shape='SPT',
                     geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'), number: int = None) \
        -> list:
    """
    Get trajectories based on the trajectories saved in myDataFrame.
    :param geometry: geometry of the maze, given by the names of the excel files with the dimensions
    :param number: How many trajectories do you want to have in your list?
    :param solver: str
    :param size: str
    :param shape: str
    :return: list of objects, that are of the class or subclass Trajectory
    """
    df = myDataFrame[
        (myDataFrame['size'] == size) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver) &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]


def plot_transition_matrix(T: np.array, state_order: list, name: str = 'transition_matrix'):
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(T)

    ax.set_xticks(range(len(state_order)))
    ax.set_yticks(range(len(state_order)))
    ax.set_xticklabels(state_order)
    ax.set_yticklabels(state_order)
    directory = graph_dir() + os.path.sep + name + '.pdf'
    print('Saving transition matrix in ', directory)
    plt.gcf().savefig(directory)


sizes = {0: 5.0, 1: 10.0, 2: 5.0}
colors = {0: 'black', 1: 'red', 2: 'blue'}


def Network2igraph(network):
    """
    Returns an igraph Graph object which represents
    the k-th layer of a multi-order graphical model.
    """
    g = igraph.Graph(directed=True)

    for e in network.edges:
        if g.vcount() == 0 or e[0] not in g.vs()["name"]:
            g.add_vertex(e[0])
        if g.vcount() == 0 or e[1] not in g.vs()["name"]:
            g.add_vertex(e[1])
        g.add_edge(e[0], e[1], weight=network.edges[e]['weight'])
    return g


def plot_network(n: pp.Network, name: str = 'network') -> None:
    """
    Save .html of the network.
    :param n: Network
    :param name: name, under which the .html will be saved
    """
    g1 = Network2igraph(n)
    visual_style = {"layout": g1.layout_auto(),
                    "vertex_label": g1.vs["name"],
                    "edge_label": g1.es["weight"],
                    "width": 3200, "height": 3200,
                    # "d3js_path": # TODO
                    # "edge_width": 1
                    "label_color": "#010101",
                    # "edge_color": "#010101"
                    "node_size": {name: sizes[len(name)] for name in g1.vs["name"]},
                    "node_color": {name: colors[len(name)] for name in g1.vs["name"]},
                    }
    directory = path.join(network_dir, name + '.html')
    print('Saving network image in ', directory)
    export_html(n, directory, **visual_style)


def pathpy_network(state_series) -> (pp.Paths, pp.Network):
    # create paths and network
    paths = create_paths(state_series)
    n = pp.Network.from_paths(paths)

    state_order = ['b', 'a', 'd', 'e', 'f', 'g', 'j']
    # state_dict = {v: n.node_to_name_map()[v] for v in state_order if v in n.node_to_name_map().keys()}
    return paths, n


def absorbing_state_analysis(T) -> np.array:
    m = transposeMatrix(T)
    m = sort(m)
    # norm = normalize(m)
    trans = num_of_transients(m)
    Q, R = decompose(m)
    P = np.vstack([np.hstack([identity(1), np.zeros([1, trans])]),
                   np.hstack([R, Q])])  # canonical form of transition matrix
    N = np.linalg.inv(identity(len(Q[-1])) - Q)  # fundamental matrix
    t = np.matmul(N, np.ones(N.shape[0]))  # expected number of steps before absorption from each steps
    B = np.matmul(N, R)  # absorption probabilities
    P_inf = np.linalg.matrix_power(P, 100)  # check whether P_inf is indeed np.array([[I, 0], [B, 0]])
    return t


def Markovian_analysis(n) -> np.array:
    # adjacency_matrix, degrees, laplacian, transition matrix and eigenvector
    A = n.adjacency_matrix().toarray()
    D = np.eye(len(n.degrees())) * np.array(n.degrees())
    L = n.laplacian_matrix(n.degrees()).toarray()
    T = n.transition_matrix().toarray()
    EV = n.leading_eigenvector(n.adjacency_matrix())
    return T


def higher_order_networks(paths):
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
