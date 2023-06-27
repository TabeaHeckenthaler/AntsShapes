import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from DataFrame.plot_dataframe import save_fig
import networkx as nx
from Analysis.PathPy.Path import Path
from tqdm import tqdm
import json
from colors import colors_state, hex_to_rgb
from networkx.classes.function import path_weight
from Directories import home
from copy import copy
import itertools
import colorsys
import os
from PIL import Image

false_connections = [('b1', 'f'), ('b2', 'f'), ('be', 'eb'), ('cg', 'g'), ('eg', 'g')]


# how can we bias the network in x direction?
# increase cost of going against the x direction.
# bias = 0.2 # stretching factor of the distances in x.

connectionMatrix = pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\ConnectionMatrix.xlsx", index_col=0, dtype=bool)
img = plt.imread(home + "\\PS_Search_Algorithms\\human_network_simulation\\cleanCSBW.png")

class HumanStateMachine:
    def __init__(self, seed=42, pattern_recognition=0.8, bias=1.3, weakening_factor=0.6):
        """

        :param seed: random seed

        :param pattern_recognition: value between 0 and 1.
         The smaller, the stronger the effect of pattern recognition.
         Value by which the 'e' -> 'f' connection is weakened once simulation went from 'ac' to 'c'.
         Determines how many simulations go to 'eg'. The smaller, the more go to 'eg'.
         Has to be larger than zero, so that a connection from 'e' -> 'f' is not totally excluded.

        :param bias: bias factor, 0 < b < 1.
         The larger this value is the more the network is biased in x direction
         Every pathlength is weighed with (1 - self.bias * np.cos(angle))/(1 + self.bias),
         where alpha is the angle between the pathד FIRST step and the bias direction.
         Determines how many simulations go to 'ac' or 'ab'. The smaller, the more go to 'ab' initially.

        :param weakening_factor: value between 0 and 1.
         The larger, the more they repeat the same mistake
         Factor by which the passage probability is reduced once a faulty connection was experienced.
         Determines the total number passages. The larger, the more passages.
        """
        self.seeds, self.gen = self.getSeeds(seed)
        self.points = self.getPoints()

        self.distances = self.calc_distances(self.points)
        self.angles = self.calc_angles(self.points)
        # save self.perceivedDistances to excel
        # self.perceivedDistances.to_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\calc_distances.xlsx")

        self.passage_probabilities = self.init_perceived_affinities()
        self.G = self.create_graph()
        self.initial_state = 'ab'
        self.target = 'i'
        self.path = []
        self.anim = None
        self.i = 0
        self.dont_go_back = []
        if 0 < pattern_recognition <= 1:
            self.pattern_recognition = pattern_recognition
        else:
            raise ValueError("0 < pattern_recognition <= 1.")

        if 0 <= bias <= 1:
            self.pattern_recognition = pattern_recognition
        else:
            raise ValueError("0 <= bias <= 1.")
        self.bias = bias
        self.weakening_factor = weakening_factor
        self.chosen_path = None

    def init_perceived_affinities(self):
        # create a matrix of the perceived connections
        small_average_initialNetworks = connectionMatrix.copy()

        for i, j in false_connections:
            small_average_initialNetworks.loc[i][j] = True
            small_average_initialNetworks.loc[j][i] = True
        return small_average_initialNetworks.astype(float)

        # # create a matrix of the perceived connections
        # small_average_initialNetworks = \
        #     pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\small_average_initialNetworks.xlsx",
        #                   index_col=0)
        # percievedConnectionMatrix = connectionMatrix.copy().astype(float)
        #
        # # passage_probabilities = pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\"
        # #                                              "passage_probabilities.xlsx",
        # #                                       index_col=0)
        #
        # for i, j in itertools.combinations(percievedConnectionMatrix.index, 2):
        #     p = small_average_initialNetworks.loc[i][j]
        #     # choice = self.gen.choice([True, False], p=[p, 1-p])
        #     percievedConnectionMatrix.loc[i][j] = p
        #     percievedConnectionMatrix.loc[j][i] = p
        # return percievedConnectionMatrix

    @property
    def nodes(self):
        return self._x

    @nodes.getter
    def nodes(self):
        return self.distances.index.tolist()

    @staticmethod
    def is_connected(i, j):
        # print([i.strip('.')], [j.strip('.')], connectionMatrix.loc[i.strip('.')][j.strip('.')])
        return connectionMatrix.loc[i.strip('.')][j.strip('.')]

    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        edge_costs = {}
        for i in self.nodes:
            for j in self.nodes:
                if self.passage_probabilities.loc[i][j] > 0:
                    G.add_edge(i, j)
                    edge_costs[(i, j)] = self.distances.loc[i][j] # / self.passage_probabilities.loc[i][j]
        nx.set_edge_attributes(G, edge_costs, 'cost')
        return G


    def getPoints(self, *args) -> list:
        # read exel file
        df = pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\points_states_in_image.xlsx", index_col=0)
        return df

    @staticmethod
    def getSeeds(rndSeed, numPoints=5):
        gen = np.random.default_rng(rndSeed)
        seeds = gen.random((numPoints, 2)) * 2 - 1
        return seeds, gen

    def shortest_path(self, source, curr_i):
        # Set the custom filter_neighbors function as the weight function for Dijkstra's algorithm
        G_copy = self.G.copy()
        G_copy.remove_node(curr_i)
        path = nx.shortest_path(G_copy, source=source, target=self.target, weight='cost')
        return path

    def path_length(self, path):
        path_length = path_weight(self.G, path=path, weight='cost')
        # shortest_path_length = nx.shortest_path_length(self.G, source=source, target=self.target, weight='cost')
        return path_length

    def path_length_biased(self, path) -> float:
        """
        :param path:
        :return: biased path length
        """
        # the smaller the path length, the larger the probability of choosing this path.
        path_length = path_weight(self.G, path=path, weight='cost')

        # shortest_path_length = nx.shortest_path_length(self.G, source=source, target=self.target, weight='cost')
        total_passage_probability = np.prod([self.passage_probabilities.loc[n1][n2]
                                             for n1, n2 in zip(path[:-1], path[1:])])
        # if total_passage_probability < 1:
        #     DEBUG = 1

        # the smaller the angle, the smaller the bias_factor, which reduces the path_length.
        # bias_factor = self.bias - np.cos(self.angles.loc[path[1]][path[0]])
        bias_factor = (1 - self.bias * np.cos(self.angles.loc[path[1]][path[0]]))/(1 + self.bias)

        # bias_factor should be 1, for all angles given a certain self.bias.
        # bias_factor should be < 1, for all angles given a certain self.bias

        return path_length * bias_factor / total_passage_probability


    def plot_network(self, G=None, paths=[], curr_i=None, labels: dict = None, attempt_i=None, bottom_text=None):
        if G is None:
            G = self.G
        pos_df = self.getPoints()
        # pos = {node: [d['x'], d['theta']] for node, d in pos_df.to_dict(orient='index').items()}
        pos = {node: [pos_df.loc[node.replace('.','')]['x'],
                      pos_df.loc[node.replace('.','')]['theta']] for node in G.nodes}
        edge_labels = {state_combo: int(dist) for state_combo, dist in nx.get_edge_attributes(G, 'cost').items()
                       if self.distances.loc[state_combo[0]][state_combo[1]] > 0}

        fig, ax = plt.subplots()
        # determine size of fig
        fig.set_size_inches(8, 10)
        im = ax.imshow(img)

        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='k')
        # nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
        if labels is not None:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_weight='bold', label_pos=0.2)

        for path in paths:
            path = [curr_i] + path
            path_edges = list(zip(path, path[1:]))
            color = hex_to_rgb(colors_state[path_edges[1][1].replace('.','')])
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color,
                                   width=3, style='dotted')
            # nx.draw_networkx_nodes(G, pos, nodelist=[path_edges[0][0]], node_color=np.array(color), node_size=300)
            nx.draw_networkx_nodes(G, pos, nodelist=[path_edges[0][1]], node_size=100, node_color=color)


        if curr_i is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[curr_i], node_color=hex_to_rgb(colors_state[curr_i.replace('.','')]), node_size=300)
            nx.draw_networkx_labels(G, pos, labels={curr_i: curr_i}, font_size=12, font_weight='bold')

        if attempt_i is not None:
            # draw a hollow black thick circle aruond the node

            plt.gca().add_artist(plt.Circle(pos[attempt_i], radius=5, color='k', fill=False, lw=20))

        if bottom_text is not None:
            # write path plan on the bottom of the image in Times New Roman
            plt.text(0.5, -0.05, bottom_text, ha='center', va='center', transform=ax.transAxes, fontsize=12,
                     fontname='Times New Roman')

        # equal aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(self.get_image_dir() + '\\' + str(self.i) + "_network.png", dpi=150)
        DEBUG = 1
        plt.close()

    def get_image_dir(self):
        dir = "\\PS_Search_Algorithms\\human_network_simulation\\images\\" + str(i)
        if not os.path.exists(home + dir):
            os.mkdir(home + dir)
        return home + dir

    @staticmethod
    def same_paths(path1, path2):
        path1 = list(map(lambda x: x.replace('b1', 'b2'), path1))
        path2 = list(map(lambda x: x.replace('b1', 'b2'), path2))

        # a path is similar if all nodes of one path are contained in the other path
        if set(path1).issubset(set(path2)) or set(path2).issubset(set(path1)):
            return True
        return False

    def calc_costs(self, paths):
        costs = dict()
        for path in paths:
            # check if there is already a path that contains exactly the same nodes
            # and no equivalent states in path and self.dont_go_back

            # if len(set(self.dont_go_back[:-2] + [self.dont_go_back[-1]]).intersection(path[1:])) == 0:
            if len(set(self.dont_go_back).intersection(path[1:])) == 0:
                if not np.any([self.same_paths(path.copy(), eval(s2)) for s2 in costs.keys()]):
                    costs[str(path)] = self.path_length_biased(path) # ** 2
                    # costs[str(path)] = self.path_length(path)

        # print(costs)
        return costs

    def choose_new_path(self):
        paths = [j for j in nx.shortest_simple_paths(self.G, source=curr_node, target=self.target, weight='cost')]
        costs = self.calc_costs(paths)

        p = [1 / cost for path, cost in costs.items()]
        # if nan in p, set to 0
        p = np.array(p) / np.sum(p)

        if np.any(np.isnan(p)):
            costs = self.calc_costs(paths)
        if len(costs) == 0:
            costs = self.calc_costs(paths)


        if np.any(np.isnan(p)):
            DEBUG = 1
            self.calc_costs(paths)
            # p = np.nan_to_num(p)
        chosen_path_str = self.gen.choice(list(costs.keys()), p=p)
        self.chosen_path = eval(chosen_path_str)
        if 'eg' in self.chosen_path:

            DEBUG = 1
            costs = self.calc_costs(paths)

    def update_state(self, log=False):
        """
        Update the state of the system
        """
        # find next node to go to according to gradient descent in graph
        global curr_node, attempt_i
        if self.chosen_path is None:
            self.choose_new_path()
        else:
            self.chosen_path = self.chosen_path[1:]
        next_node = self.chosen_path[1]

        if self.is_connected(curr_node, next_node):
            self.strengthen_connection(curr_node, next_node)
            curr_node = copy(next_node)
            self.path.append(curr_node)
        else:
            self.weaken_connection(curr_node, next_node, connection=False)
            self.chosen_path = None
        return curr_node


    def update_state_no_return(self, log=False):
        """
        Update the state of the system
        """
        # find next node to go to according to gradient descent in graph
        global curr_node, attempt_i

        self.dont_go_back.append(curr_node)
        self.choose_new_path()

        next_node = chosen_path[1]

        if self.is_connected(curr_node, next_node):
            if log:
                to_log += "\n" + curr_node + ' ====> ' + next_node

            self.strengthen_connection(curr_node, next_node)
            curr_node = copy(next_node)
            self.path.append(curr_node)

        else:
            if log:
                to_log += "\n" + curr_node + ' ==/==> ' + next_node

            self.weaken_connection(curr_node, next_node, connection=False)
            self.dont_go_back = [next_node]

        # if not os.path.exists('images\\' + str(i)):
        #     os.mkdir('images\\' + str(i))
        #
        if log:
            with open('images\\' + str(i) +'\\log.txt', 'a') as f:
                f.write(to_log)
                f.close()

        return curr_node

    def there_is_another_node(self, name):
        num_existing_nodes = len([n for n in self.nodes if n.strip('.') == name.strip('.')])

        # draw from exponential distribution
        if num_existing_nodes == 0:
            raise ValueError('there is no node with name', name)
        else:
            lamb = 0.8 # the higher this value the more likely it is to add a new node
            return self.gen.uniform(0, 1) < 1 / (num_existing_nodes + 1)

    def add_new_node(self, name):
        new_name = name + '.'
        # concate position to self.points with index new_name
        row_to_duplicate = self.points.loc[name]
        self.points = pd.concat([self.points, row_to_duplicate.to_frame().T], axis=0)
        self.points.index = pd.Index(list(self.points.index[:-1]) + [new_name])

        row_to_duplicate = self.distances.loc[name]
        self.distances = pd.concat([self.distances, row_to_duplicate.to_frame().T], axis=0)
        self.distances.index = pd.Index(list(self.distances.index[:-1]) + [new_name])

        column_to_duplicate = self.distances.loc[name]
        # append to column_to_duplicate a row with index new_name
        column_to_duplicate = pd.concat([column_to_duplicate, pd.Series([np.inf], index=[new_name])], axis=0)

        self.distances = pd.concat([self.distances, column_to_duplicate.to_frame()], axis=1)
        self.distances.columns = pd.Index(list(self.distances.columns[:-1]) + [new_name])

    def run(self, cutoff=100):
        global curr_node
        curr_node = self.initial_state
        self.path = [curr_node]
        while curr_node != self.target and self.i < cutoff:
            # curr_node = self.update_state_no_return()
            curr_node = self.update_state()
            self.i += 1

    @staticmethod
    def save(paths, name):
        # name = 'pattern_recognition_' + str(pattern_recognition) + '_ineg_bias_' + str(b) + \
        #        '_weakening_factor_' + str(self.weakening_factor)

        with open('simulation_results\\' + name + "_states_small.json", 'w') as f:
            json.dump(paths, f)

    def animate(self): # plt.style.use('dark_background')
        fig, ax = plt.subplots()
        im = ax.imshow(img)

        # ax.plot(self.seeds[:, 0], self.seeds[:, 1], 'ro')
        points_plot = ax.plot(self.points['x'], self.points['y'], 'ko')[0]

        dude_plot = ax.plot([self.points.loc[self.initial_state, 'x']],
                            [self.points.loc[self.initial_state, 'y']],
                            'ro', markersize=10)[0]

        global curr_node, last_i, attempt_i
        curr_node = self.initial_state
        attempt_i = self.initial_state

        self.paths = [[curr_node]]
        plt.tight_layout()

        def other_update_anim(frame):
            self.update_state()
            dude_plot.set_data([self.points.loc[curr_node, 'x']],
                               [self.points.loc[curr_node, 'y']])
            # add a line to ax
            ax.plot([self.points.loc[last_i, 'x'], self.points.loc[curr_node, 'x']],
                    [self.points.loc[last_i, 'y'], self.points.loc[curr_node, 'y']], 'k-')

            # add a line to ax
            if attempt_i != curr_node:
                ax.plot([self.points.loc[attempt_i, 'x'], self.points.loc[curr_node, 'x']],
                        [self.points.loc[attempt_i, 'y'], self.points.loc[curr_node, 'y']], 'r-')

            return dude_plot

        self.anim = animation.FuncAnimation(fig, other_update_anim, interval=400, repeat=False, frames=range(40))
        plt.show()

    @staticmethod
    def calc_angles(points):
        l = points.shape[0]
        points_array = np.array(points)
        deltas = np.repeat(points_array, l, 0).reshape((l, l, 3))
        deltas -= deltas.swapaxes(0, 1)
        dists = pd.DataFrame(np.linalg.norm(deltas, axis=-1), index=points.index, columns=points.index)

        # calculate the angle between the vector from i to j and the x axis
        with np.errstate(divide='ignore', invalid='ignore'):
            angles = pd.DataFrame(np.arccos(deltas[:, :, 0]/np.linalg.norm(deltas, axis=-1)),
                                  index = points.index, columns = points.index)
        return angles

    @staticmethod
    def calc_distances(points):
        l = points.shape[0]
        points_array = np.array(points)
        deltas = np.repeat(points_array, l, 0).reshape((l, l, 3))
        deltas -= deltas.swapaxes(0, 1)

        # deltas[:, :, 0] = deltas[:, :, 0] * bias
        # print('biased deltas with factor ', bias, ' in x axis')
        dists = pd.DataFrame(np.linalg.norm(deltas, axis=-1), index=points.index, columns=points.index)

        # against_bias = deltas.copy()
        # against_bias[:, :, 0] = against_bias[:, :, 0] / bias
        # with_bias = deltas.copy()
        # with_bias[:, :, 0] = with_bias[:, :, 0] * bias
        #
        # dists_against_bias = pd.DataFrame(np.linalg.norm(against_bias, axis=-1), index=points.index, columns=points.index)
        # dists_with_bias = pd.DataFrame(np.linalg.norm(with_bias, axis=-1), index=points.index, columns=points.index)

        # # iterate over all nodes
        # for i, j in itertools.combinations(dists_against_bias.index, 2):
        #     # if i has a smaller x value than j use the distance with bias
        #     if points.loc[i, 'x'] < points.loc[j, 'x']:
        #         dists_against_bias.loc[i, j] = dists_with_bias.loc[i, j]
        # dists = dists_against_bias

        dists.loc['f']['h'] = dists.loc['g']['h']
        dists.loc['h']['f'] = dists.loc['g']['h']
        return dists

    @staticmethod
    def plot_bar_chart(paths):
        fig, ax = plt.subplots()
        for i, path in enumerate(paths):
            p = Path(time_step=60, time_series=path)
            p.bar_chart(ax=ax, axis_label=str(i), array=[state.strip('.') for state in path[:-1]], block=True)
        plt.subplots_adjust(hspace=.0)
        save_fig(fig, name='human_simulation_bar_chart_no_pattern')

    @classmethod
    def df(cls, name='states_small.json') -> pd.DataFrame:
        """
        Load simulated data and return as a dataframe
        :param name: name of the json file
        :return: A dataframe with the columns: name, size, winner, states_series
        """
        with open(home + "\\PS_Search_Algorithms\\human_network_simulation\\simulation_results\\" + name, 'r') as f:
            paths = json.load(f)
        # create dataframe with colums: name,
        tuples = [(name, 'Small', True, path) for name, path in enumerate(paths)]
        df = pd.DataFrame(tuples, columns=['name', 'size', 'winner', 'states_series'])
        return df

    def update_connection_old(self, state1, state2, connection=False, add_new_node=False):
        if add_new_node:
            self.add_new_node(curr_i)
        self.distances.loc[state1, state2] = np.inf
        self.distances.loc[state2, state1] = np.inf
        self.G = self.create_graph()

    def strengthen_connection(self, state1, state2):
        if (state1 == 'ac' and state2 == 'c' or state1 == 'c' and state2 == 'ac'):
            self.passage_probabilities.loc['f', 'e'] = self.pattern_recognition * self.passage_probabilities.loc['f', 'e']
            self.passage_probabilities.loc['e', 'f'] = self.pattern_recognition * self.passage_probabilities.loc['e', 'f']

            self.G['f']['e']['cost'] = self.distances.loc['f', 'e'] / self.passage_probabilities.loc['f', 'e']
            self.G['e']['f']['cost'] = self.distances.loc['e', 'f'] / self.passage_probabilities.loc['e', 'f']
            self.chosen_path = None
            #
            # self.passage_probabilities.loc['eg', 'g'] = self.pattern_recognition * self.passage_probabilities.loc['eg', 'g']
            # self.passage_probabilities.loc['g', 'eg'] = self.pattern_recognition * self.passage_probabilities.loc['g', 'eg']

            # # check whether edge between 'eg' and 'g' are in the graph
            # if ('eg', 'g') in self.G.edges() and ('g', 'eg') in self.G.edges():
            #     self.G['eg']['g']['cost'] = self.distances.loc['eg', 'g'] / self.passage_probabilities.loc['eg', 'g']
            #     self.G['g']['eg']['cost'] = self.distances.loc['g', 'eg'] / self.passage_probabilities.loc['g', 'eg']

        self.passage_probabilities.loc[state1, state2] = 1
        self.passage_probabilities.loc[state2, state1] = 1

        self.G[state1][state2]['cost'] = \
            self.distances.loc[state1, state2] / \
            self.passage_probabilities.loc[state1, state2]

    def weaken_connection(self, state1, state2, connection=False):
        if (state1 == 'eg' and state2 == 'g'):
            self.passage_probabilities.loc['f', 'e'] = 1 / self.pattern_recognition * self.passage_probabilities.loc['f', 'e']
            self.passage_probabilities.loc['e', 'f'] = 1 / self.pattern_recognition * self.passage_probabilities.loc['e', 'f']

            self.G['f']['e']['cost'] = self.distances.loc['f', 'e'] / self.passage_probabilities.loc['f', 'e']
            self.G['e']['f']['cost'] = self.distances.loc['e', 'f'] / self.passage_probabilities.loc['e', 'f']

        # if add_new_node:
        #     self.add_new_node(curr_i)
        # self.passage_probabilities.loc[state2, state1] = self.passage_probabilities.loc[state2, state1] / 2
        # self.passage_probabilities.loc[state1, state2] = self.passage_probabilities.loc[state1, state2] / 2
        if state1 == 'cg':
            DEBUG = 1

        self.passage_probabilities.loc[state1, state2] = \
            self.passage_probabilities.loc[state1, state2] * self.weakening_factor
        self.passage_probabilities.loc[state2, state1] = \
            self.passage_probabilities.loc[state2, state1] * self.weakening_factor

        if self.passage_probabilities.loc[state1, state2] == 0:
            self.G.remove_edge(state1, state2)
            # self.G[state1][state2]['cost'] = np.inf
            # self.G[state2][state1]['cost'] = np.inf
        else:
            self.G[state1][state2]['cost'] = \
                self.distances.loc[state1, state2] / self.passage_probabilities.loc[state1, state2]
            self.G[state2][state1]['cost'] = \
                self.distances.loc[state2, state1] / self.passage_probabilities.loc[state2, state1]

    def merge(self):
        # merge images
        dir_path = self.get_image_dir()
        # Create a list of image file paths
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')]

        # Open and resize all images to the same height
        # scale = 1 / len(file_paths)
        scale = 1
        images_with_white = [Image.open(fp).resize((int(Image.open(fp).width / scale), Image.open(fp).height)) for fp in
                             file_paths if fp != dir_path + '\\all.png']

        images = []
        for image in images_with_white:
            # cut out the white space
            # get bounding box of area that is not white
            not_white = np.where(np.array(image) != 255)
            # get the bounding box
            bbox = np.min(not_white[1]), np.min(not_white[0]), np.max(not_white[1]), np.max(not_white[0])
            images.append(image.crop(bbox))

        # Calculate the total width of the merged image
        total_width = sum([img.width for img in images])

        # Create a new blank image with the required size
        result = Image.new('RGB', (total_width, images[0].height))

        # Merge all images horizontally into the new image
        x_offset = 0
        for img in images:
            result.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the merged image as a new PNG file
        result.save(self.get_image_dir() + '\\all.png')


if __name__ == '__main__':
    # initial_Markovian_Matrix = pd.read_excel("initialMarkovianMatrix.xlsx", index_col=0)
    # percievedConnectionMatrix = pd.read_excel("PercievedConnectionMatrix.xlsx", index_col=0)


    df = HumanStateMachine.df()
    paths = []
    for i in tqdm(range(60)):
        # i = 62
        stateMachine = HumanStateMachine(seed=i)
        stateMachine.run()
        paths.append(stateMachine.path)
        # stateMachine.merge()
        DEBUG = 1

    HumanStateMachine.save(paths)
    # stateMachine.animate()
    # print(stateMachine.paths)

    # Can we capture more of the dynamics ... for example movement inside of the state?
    # TODO: plot the time spent in each state -> distribution
    # TODO: Are the long time spans just a lot of waiting?
    # TODO: if we wanted to capture waiting...
