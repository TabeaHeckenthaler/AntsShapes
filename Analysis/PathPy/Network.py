import pathpy
from pathpy.visualisation import export_html
from matplotlib import pyplot as plt
import igraph
from Directories import network_dir
from trajectory_inheritance.exp_types import exp_types
import os
from copy import copy
import json
from Analysis.GeneralFunctions import graph_dir
from Analysis.PathPy.Paths import PathWithoutSelfLoops
from Analysis.PathPy.SPT_states import final_state
import pandas as pd
import numpy as np

sizes = {0: 5.0, 1: 10.0, 2: 5.0}
colors = {0: 'black', 1: 'red', 2: 'blue'}


class Network(pathpy.Network):

    # def __init__(self, solver, size, shape):
    #     super().__init__()
    #     if 'Small' in size:
    #         size = 'Small'
    #     self.name = '_'.join(['network', solver, size, shape])
    #     self.paths = None
    #     self.T = None  # transition matrix
    #     self.N = None  # fundamental matrix
    #     self.Q = None
    #     self.t = None  # absorption time
    #     self.R = None
    #     self.P = None  # canonical form of transition matrix
    #     self.B = None
    #
    #     # if possible_transitions is not None:
    #     #     for state1, state2 in itertools.product(possible_transitions, possible_transitions):
    #     #         self.add_edge(state1, state2, weight=0)

    @classmethod
    def init_from_paths(cls, paths, solver, size, shape):
        self = super(Network, cls).from_paths(paths)
        if 'Small' in size:
            size = 'Small'
        self.name = '_'.join(['network', solver, size, shape])
        self.paths = paths
        if type(self.paths) is PathWithoutSelfLoops:
            self.name = self.name + '_state_transitions'
        self.add_edges()
        self.T = None  # transition matrix
        self.N = None  # fundamental matrix
        self.Q = None
        self.t = None  # absorption time
        self.R = None
        self.P = None  # canonical form of transition matrix
        self.B = None
        return self

    def to_dict(self) -> dict:
        if self.N is None:
            self.markovian_analysis()

        return {'T': self.T.to_json(), 'N': self.N.to_json(), 'Q': self.Q.to_json(), 't': self.t.to_json(),
                'R': self.R.to_json(), 'P': self.P.to_json(), 'B': self.B.to_json()}

    def save_dir_results(self):
        return os.path.join(network_dir, 'MarkovianNetworks', self.name + '.txt')

    def save(self, results):
        with open(self.save_dir_results(), 'w') as json_file:
            json.dump(results, json_file)
            print('Saved Markovian results in ', self.save_dir_results())

    def get_results(self):
        if os.path.exists(self.save_dir_results()):
            with open(self.save_dir_results(), 'r') as json_file:
                attribute_dict = json.load(json_file)
            self.T = pd.read_json(attribute_dict['T'])
            self.N = pd.read_json(attribute_dict['N'])
            self.Q = pd.read_json(attribute_dict['Q'])
            self.t = pd.read_json(attribute_dict['t'], typ='series')
            self.R = pd.read_json(attribute_dict['R'])
            self.P = pd.read_json(attribute_dict['P'])
            self.B = pd.read_json(attribute_dict['B'])
        else:
            self.markovian_analysis()
            # self.save_results(self.to_dict())

    def add_edges(self):
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
        directory = os.path.join(network_dir, 'Network_Images', self.name + '.html')
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

    def plot_transition_matrix(self, title: str = '', axis=None, state_order=None):
        if self.T is None:
            self.markovian_analysis()
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        if state_order is not None:
            for state in state_order:
                if state not in self.T.columns:
                    self.T[state] = np.NaN
            to_plot = self.T.reindex(state_order)[state_order]
        else:
            to_plot = self.T
        _ = axis.imshow(to_plot)

        axis.set_xticks(range(len(to_plot)))
        axis.set_xticklabels(to_plot.columns)

        axis.set_yticks(range(len(to_plot)))
        axis.set_yticklabels(to_plot.columns)
        axis.set_title(title)

    @staticmethod
    def swap(m, row2, row1):
        order = m.index.tolist()
        order[row2], order[row1] = copy(order[row1]), copy(order[row2])
        m = m.reindex(columns=order, index=order)
        return m

    def find_P(self, T):
        """
        reorganize matrix to canonical form
        """
        num_absorbing = 0
        for i, r in enumerate(T.index):
            if 1.0 == T.loc[r][r]:
                num_absorbing += 1
                for ii in range(num_absorbing, i + 1)[::-1]:
                    T = self.swap(T, ii, ii - 1)
        return T, num_absorbing

    def markovian_analysis(self):
        # self.T = pd.DataFrame([[1, 0, 0, 0, 0], [1/2, 0, 1/2, 0, 0], [0, 1/2, 0, 1/2, 0],
        #                        [0, 0, 1/2, 0, 1/2], [0, 0, 0, 0, 1]],
        #                       columns='0,1,2,3,4'.split(','),
        #                       index='0,1,2,3,4'.split(','))

        self.T = pd.DataFrame(self.transition_matrix().toarray().transpose(),
                              columns=list(self.node_to_name_map()),
                              index=list(self.node_to_name_map()))
        self.T[final_state][final_state] = 1
        self.P, num_absorbing = self.find_P(self.T)
        self.Q, self.R = self.P.iloc[num_absorbing:, num_absorbing:], self.P.iloc[num_absorbing:, 0:num_absorbing]
        transient_state_order = self.P.columns[num_absorbing:]
        self.N = pd.DataFrame(np.linalg.inv(np.identity(self.Q.shape[-1]) - self.Q),
                              columns=transient_state_order,
                              index=transient_state_order
                              )  # fundamental matrix
        self.B = pd.DataFrame(np.matmul(self.N.to_numpy(), self.R.to_numpy()),
                              index=transient_state_order,
                              columns=self.T.index[-num_absorbing:]
                              )  # absorption probabilities
        self.t = np.matmul(self.N, np.ones(self.N.shape[0]))

    def create_higher_order_network(self, k=2, null_model=True) -> pathpy.Network:
        hon = pathpy.HigherOrderNetwork(self.paths, k=k, null_model=null_model)
        # for e in hon.edges:
        #     print(e, hon.edges[e])
        return hon

    def diffusion_speed_up(self):
        """create higher order model.
        1. T: where each node represents a k length subpath
        2. T_tag: based on the 1st order model which in general allows different transitions
        """
        hon = self.create_higher_order_network(k=2, null_model=False)
        hon_null = self.create_higher_order_network(k=2, null_model=True)

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
        return S

    def estimate_memory(self):
        """
        Actually https://www.pathpy.net/tutorial/model_selection.html describes how to select the best higher order
        model. Its not via the diffusion speedup/slow down but via statistical likelihood. They already have the
        estimate of the order. Seems that for both cases best order that describes the data is second order model.
        """
        p = my_network.paths
        mog = pathpy.MultiOrderModel(p, max_order=10)
        print('Optimal order = ', mog.estimate_order())


# def create_higher_order_network(self, k: int = 2) -> pp.Network:
#     hon = pp.HigherOrderNetwork(self.paths, k=k, null_model=True)
#     # for e in hon.edges:
#     #     print(e, hon.edges[e])
#     return hon
#
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
#     # estimate the order of the sequence (calculate_diffusion_time like memory)
#     ms = pp.MarkovSequence(self.paths.sequence())
#     order = ms.estimate_order(4)


def plot_diffusion_time(networks):
    for my_network in networks:
        if i == 0:
            t = my_network.t.sort_values(0, ascending=False)
            index = t.index
            ax.set_xticks(ticks=range(len(index)))
            ax.set_xticklabels(index)
        else:
            # t = my_network.t.reindex(index)
            t = my_network.t.loc[index]

        t.plot(ax=ax, label=size)

    plt.show(block=False)
    ax.set_ylabel('humans: number of states to pass before solving')
    ax.legend(exp_types[shape][solver])
    fig.savefig(os.path.join(graph_dir(), 'diffusion_time_' + solver + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')


if __name__ == '__main__':
    shape = 'SPT'
    solvers = ['human', 'ant']
    geometries = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                  'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')}
    sizes = {'ant': ['XL', 'L', 'M', 'S'], 'human': ['Large', 'Medium', 'Small Far']}
    figures = {solver: plt.subplots(1, len(sizes[solver])) for solver in solvers}

    index = None
    networks = []
    for solver in list(geometries.keys()):
        fig, axs = figures[solver]
        for size, ax in zip(sizes[solver], axs):
            paths = PathWithoutSelfLoops(solver, size, shape, geometries[solver])
            paths.load_paths()
            my_network = Network.init_from_paths(paths, solver, size, shape)
            my_network.get_results()
            networks.append(my_network)

    plot_diffusion_time(networks)
    DEBUG = 1
