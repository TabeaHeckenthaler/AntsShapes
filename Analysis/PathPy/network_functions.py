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
from copy import copy
import itertools
import json


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
        (myDataFrame['initial condition'] == 'back') &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]


sizes = {0: 5.0, 1: 10.0, 2: 5.0}
colors = {0: 'black', 1: 'red', 2: 'blue'}


class Network(pp.Network):
    def __init__(self, possible_transitions: list, name='network'):
        super().__init__(directed=True)
        self.possible_transitions = possible_transitions
        self.name = name
        self.paths = pp.Paths()
        for state1, state2 in itertools.product(possible_transitions, possible_transitions):
            self.add_edge(state1, state2, weight=0)
        self.N = None
        self.Q = None
        self.t = None
        self.R = None
        self.P = None
        self.B = None

    def save_dir(self):
        return self.name + '_N.txt'

    def save_fundamental_matrix(self):
        if self.N is None:
            self.calc_fundamental_matrix()

        with open(self.save_dir(), 'w') as json_file:
            json.dump(self.N.tolist(), json_file)

    def get_fundamental_matrix(self):
        if not os.path.exists(self.save_dir()):
            raise ValueError('You havent saved the fundamental_matrix!')
        with open(self.save_dir(), 'r') as json_file:
            self.N = json.load(json_file)

    def add_paths(self, transitions: list) -> None:
        [self.paths.add_path(transition) for transition in transitions]

        for states, weight in self.paths.paths[1].items():
            if len(states) == 2:
                self.add_edge(states[0], states[1], weight=weight[0])

    def plot_network(self):
        n_to_plot = self.reduced_network()
        g1 = n_to_plot.iGraph()
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
        directory = path.join(network_dir, self.name + '.html')
        print('Saving network image in ', directory)
        export_html(n_to_plot, directory, **visual_style)

    def reduced_network(self):
        """
        Reduce edges to edges with weight larger than 0.
        :return: network
        """
        n = copy(self)

        to_remove = []
        for edge, weight_dict in n.edges.items():
            if weight_dict['weight'] < 1:
                to_remove.append(edge)

        for edge in to_remove:
            n.remove_edge(edge[0], edge[1])
        return n

    def iGraph(self) -> igraph.Graph:
        """
        Returns an igraph Graph object which represents
        the k-th layer of a multi-order graphical model.
        """
        g = igraph.Graph(directed=True)
        for e in self.edges:
            if g.vcount() == 0 or e[0] not in g.vs()["name"]:
                g.add_vertex(e[0])
            if g.vcount() == 0 or e[1] not in g.vs()["name"]:
                g.add_vertex(e[1])
            g.add_edge(e[0], e[1], weight=self.edges[e]['weight'])
        return g

    def plot_transition_matrix(self, T: np.array = None, title: str = '', axis=None):
        if T is None:
            T = self.transition_matrix().toarray()
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        _ = axis.imshow(T)
        axis.set_xticks(range(len(self.possible_transitions)))
        axis.set_yticks(range(len(self.possible_transitions)))
        axis.set_xticklabels(self.possible_transitions)
        axis.set_yticklabels(self.possible_transitions)
        directory = graph_dir() + os.path.sep + 'T_' + self.name + '.pdf'
        print('Saving transition matrix in ', directory)
        plt.gcf().savefig(directory)
        plt.title(title)

    def calc_fundamental_matrix(self):
        T = self.transition_matrix().toarray()
        m = transposeMatrix(T)
        m = sort(m)
        # norm = normalize(m)
        trans = num_of_transients(m)
        self.Q, self.R = decompose(m)
        self.P = np.vstack([np.hstack([identity(1), np.zeros([1, trans])]),
                            np.hstack([self.R, self.Q])])  # canonical form of transition matrix
        P_inf = np.linalg.matrix_power(self.P, 100)  # check whether P_inf is indeed np.array([[I, 0], [B, 0]])
        print(P_inf)
        self.N = np.linalg.inv(identity(len(self.Q[-1])) - self.Q)  # fundamental matrix

    def calc_expected_absorption_time(self):
        if self.N is None:
            self.calc_fundamental_matrix()
        self.t = np.matmul(self.N,
                           np.ones(self.N.shape[0]))  # expected number of steps before absorption from each steps

    def calc_absorption_probabilities(self):
        if self.N is None:
            self.calc_fundamental_matrix()

    def absorbing_state_analysis(self):
        if self.N is None:
            self.calc_expected_absorption_time()
        self.B = np.matmul(self.N, self.R)  # absorption probabilities

    # def create_higher_order_network(self, k: int = 2) -> pp.Network:
    #     hon = pp.HigherOrderNetwork(self.paths, k=k, null_model=True)
    #     # for e in hon.edges:
    #     #     print(e, hon.edges[e])
    #     return hon
    #

    # def Markovian_analysis(self) -> np.array:
    #     # adjacency_matrix, degrees, laplacian, transition matrix and eigenvector
    #     A = self.adjacency_matrix().toarray()
    #     D = np.eye(len(self.degrees())) * np.array(self.degrees())
    #     L = self.laplacian_matrix(self.degrees()).toarray()
    #     T = self.transition_matrix().toarray()
    #     EV = self.leading_eigenvector(self.adjacency_matrix())
    #     return T
    #
    # def higher_order_networks(self):
    #     hon = self.create_higher_order_network()
    #     hon.likelihood(self.paths)'
    #     eigenvalue_gap = pp.algorithms.spectral.eigenvalue_gap(hon)
    #
    #     # How much slower is the second order network than the markovian network?
    #     pp.algorithms.path_measures.slow_down_factor(self.paths)
    #
    #     # slow down factor
    #     # Ratios smaller than one indicate that the temporal network exhibits non - Markovian characteristics
    #     pp.algorithms.path_measures.entropy_growth_rate_ratio(self.paths, method='Miller')
    #
    #     # Is the process Markovian?
    #     # estimate the order of the sequence (something like memory)
    #     ms = pp.MarkovSequence(self.paths.sequence())
    #     order = ms.estimate_order(4)
