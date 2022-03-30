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
from Analysis.PathPy.AbsorbingMarkovChain import *
from Analysis.PathPy.Paths import Paths, PathsTimeStamped, PathWithoutSelfLoops

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
            self.t = pd.read_json(attribute_dict['t'])
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

    def plot_transition_matrix(self, title: str = '', axis=None):
        if self.T is None:
            self.markovian_analysis()
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        _ = axis.imshow(self.T)
        plt.xticks(range(len(self.T)), self.T.columns)
        plt.yticks(range(len(self.T)), self.T.columns)
        directory = graph_dir() + os.path.sep + 'transition_matrix_' + self.name + '.pdf'
        print('Saving transition matrix in ', directory)
        plt.gcf().savefig(directory)
        plt.title(title)

    @staticmethod
    def swap(m, row2, row1):
        order = m.index.tolist()
        order[row2], order[row1] = copy(order[row1]), copy(order[row2])
        m = m.reindex(columns=order, index=order)
        return m

    def find_canonical_from(self, T):
        """
        reorganize matrix so zero-rows go last (preserving zero rows order)
        """
        absorbing = 1
        for i, r in enumerate(T.index):
            if 1.0 == T.loc[r][r]:
                for ii in range(absorbing, i+1)[::-1]:
                    T = self.swap(T, ii, ii-1)
                absorbing += 1
        return T

    def markovian_analysis(self):
        self.T = pd.DataFrame(self.transition_matrix().toarray(),
                              columns=list(self.node_to_name_map()),
                              index=list(self.node_to_name_map()))
        self.T['j']['j'] = 1
        sorted_transition_matrix = self.find_canonical_from(self.T.transpose())
        # self.T = pd.DataFrame(normalize(sorted_transition_matrix.to_numpy()),
        #                       columns=sorted_transition_matrix.columns,
        #                       index=sorted_transition_matrix.columns)

        number_of_transient_states = num_of_transients(self.T)
        number_of_transient_states = 3
        number_of_absorbing_states = self.T.shape[0] - number_of_transient_states
        transient_state_order = sorted_transition_matrix.columns[:number_of_transient_states]

        self.Q, self.R = decompose(self.T)
        self.R.set_index(transient_state_order, inplace=True)
        self.Q.set_index(transient_state_order, inplace=True)
        self.Q.set_axis(transient_state_order, inplace=True, axis=1)

        columns = sorted_transition_matrix.columns[number_of_transient_states:].tolist() + \
                  sorted_transition_matrix.columns[:number_of_transient_states].tolist()

        self.P = pd.DataFrame(np.vstack([np.hstack([identity(number_of_absorbing_states),
                                         np.zeros([number_of_absorbing_states, number_of_transient_states])]),
                              np.hstack([self.R, self.Q])]),
                              columns=columns, index=columns)
        self.N = pd.DataFrame(np.linalg.inv(identity(self.Q.shape[-1]) - self.Q),
                              columns=transient_state_order,
                              index=transient_state_order)  # fundamental matrix
        self.B = pd.DataFrame(np.matmul(self.N.to_numpy(), self.R.to_numpy()),
                              index=transient_state_order,
                              columns=self.T.index[-number_of_absorbing_states:])  # absorption probabilities
        self.t = np.matmul(self.N, np.ones(self.N.shape[0]))  # TODO: There is something really off here, with the ant state transitions.

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

#
# class SelfLoopingNetwork(Network):
#     def __init__(self, solver: str, size: str, shape: str, paths: Union[Paths, None]):
#         super().__init__(solver, size, shape, paths)
#


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
    fig.savefig(os.path.join(graph_dir(), 'human_expected_solving_time_no_self' + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')


if __name__ == '__main__':
    # solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    solver, shape, geometry = 'ant', 'SPT', ('MazeDimensions_new2021_SPT_ant.xlsx',
                                             'LoadDimensions_new2021_SPT_ant.xlsx')
    # nodes = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)
    fig, ax = plt.subplots()
    index = None
    networks = []
    for i, size in enumerate(exp_types[shape][solver]):
        paths = PathWithoutSelfLoops(solver, size, shape, geometry)
        paths.load_paths()
        my_network = Network.init_from_paths(paths, solver, size, shape)
        my_network.get_results()
        my_network.plot_transition_matrix()
        networks.append(my_network)
        my_network.save(my_network.to_dict())

    plot_diffusion_time(networks)
    DEBUG = 1
