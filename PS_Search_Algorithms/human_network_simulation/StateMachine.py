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
from Directories import home
from copy import copy
import itertools
import colorsys
import os
from PIL import Image

connectionMatrix = pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\ConnectionMatrix.xlsx", index_col=0, dtype=bool)
img = plt.imread(home + "\\PS_Search_Algorithms\\human_network_simulation\\cleanCSBW.png")
class HumanStateMachine:
    def __init__(self, seed=42):
        self.seeds, self.gen = self.getSeeds(seed)
        self.points = self.getPoints()
        self.initial_perceived_connection_matrix = self.init_perceivedMatrix()
        self.perceivedDistances = self.distances(self.points)
        self.perceivedDistances[~self.initial_perceived_connection_matrix] = np.inf
        self.G = self.create_graph()
        self.initial_state = 'ab'
        self.final_node = 'i'
        self.path = []
        self.anim = None
        self.i = 0

    def init_perceivedMatrix(self):
        # create a matrix of the perceived connections
        small_average_initialNetworks = \
            pd.read_excel(home + "\\PS_Search_Algorithms\\human_network_simulation\\small_average_initialNetworks.xlsx",
                          index_col=0)
        percievedConnectionMatrix = connectionMatrix.copy()

        for i, j in itertools.combinations(percievedConnectionMatrix.index, 2):
            p = small_average_initialNetworks.loc[i][j]
            choice = self.gen.choice([True, False], p=[p, 1-p])
            percievedConnectionMatrix.loc[i][j] = choice
            percievedConnectionMatrix.loc[j][i] = choice
        return percievedConnectionMatrix

    @property
    def nodes(self):
        return self._x

    @nodes.getter
    def nodes(self):
        return self.perceivedDistances.index.tolist()

    @staticmethod
    def is_connected(i, j):
        # print([i.strip('.')], [j.strip('.')], connectionMatrix.loc[i.strip('.')][j.strip('.')])
        return connectionMatrix.loc[i.strip('.')][j.strip('.')]

    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_weighted_edges_from([(i, j, self.perceivedDistances.loc[i][j]) for i in self.nodes for j in self.nodes
                                   if self.perceivedDistances.loc[i][j] < np.inf])
        edge_costs = {}
        for i in self.nodes:
            for j in self.nodes:
                edge_costs[(i, j)] = self.perceivedDistances.loc[i][j]
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

    def shortest_path(self, curr_i):
        shortest_path = nx.shortest_path(self.G, source=curr_i, target=self.final_node, weight='cost')
        return shortest_path

    def shortest_path_length(self, source):
        shortest_path_length = nx.shortest_path_length(self.G, source=source, target=self.final_node, weight='cost')
        return shortest_path_length

    def plot_network(self, G=None, paths=[], curr_i=None, labels: dict = None):
        if G is None:
            G = self.G
        pos_df = self.getPoints()
        # pos = {node: [d['x'], d['theta']] for node, d in pos_df.to_dict(orient='index').items()}
        pos = {node: [pos_df.loc[node.replace('.','')]['x'],
                      pos_df.loc[node.replace('.','')]['theta']] for node in G.nodes}
        edge_labels = {state_combo: int(dist) for state_combo, dist in nx.get_edge_attributes(G, 'cost').items()}

        fig, ax = plt.subplots()
        # determine size of fig
        fig.set_size_inches(8, 10)
        im = ax.imshow(img)

        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='k')
        # nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_weight='bold', label_pos=0.2)

        for path in paths:
            path = [curr_i] + path
            path_edges = list(zip(path, path[1:]))
            color = hex_to_rgb(colors_state[path_edges[0][1].replace('.','')])
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color,
                                   width=3, style='dotted')
            # nx.draw_networkx_nodes(G, pos, nodelist=[path_edges[0][0]], node_color=np.array(color), node_size=300)
            nx.draw_networkx_nodes(G, pos, nodelist=[path_edges[0][1]], node_size=100, node_color=color)


        if curr_i is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[curr_i], node_color=hex_to_rgb(colors_state[curr_i.replace('.','')]), node_size=300)
            nx.draw_networkx_labels(G, pos, labels={curr_i: curr_i}, font_size=12, font_weight='bold')

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

    def update_state(self):
        """
        Update the state of the system
        """
        # find next node to go to according to gradient descent in graph
        global curr_i, last_i, attempt_i
        # create copy of curr_i
        last_i = copy(curr_i)

        shortest_paths = dict()
        for option in list(self.G.neighbors(curr_i)):
            shortest_paths[option] = (self.shortest_path(option), self.shortest_path_length(option))

        print(shortest_paths)
        p = [1 / v[1] for v in shortest_paths.values()]/sum([1 / v[1] for v in shortest_paths.values()])

        paths_to_highlight = [shortest_paths[attempt_i][0] for attempt_i in shortest_paths.keys()]
        path_edges_with_labels = {(curr_i, state): round(chance, 2) for state, chance in zip(shortest_paths.keys(), p)}

        attempt_i = self.gen.choice(list(shortest_paths.keys()), p=p)
        path_plan = shortest_paths[attempt_i][0]
        print(path_plan)

        self.plot_network(G=self.G, paths=paths_to_highlight, curr_i=curr_i, labels=path_edges_with_labels)

        if curr_i != 'h':
            in_plan = True
            for attempted_next_state in path_plan:
                if in_plan:
                    if self.is_connected(curr_i, attempted_next_state):
                        if curr_i == 'ac' and attempted_next_state == 'c':
                            # get edge cost for this connection
                            self.perceivedDistances.loc['e', 'f'] = self.perceivedDistances.loc['e', 'f'] * 2
                            self.perceivedDistances.loc['f', 'e'] = self.perceivedDistances.loc['f', 'e'] * 2

                            edge_cost = self.G.edges[('e', 'f')]['cost']
                            nx.set_edge_attributes(self.G, {('e', 'f'): edge_cost * 2}, 'cost')

                            in_plan = False
                            print('pattern recognition')
                        print(curr_i, ' ====> ', attempted_next_state)
                        curr_i = copy(attempted_next_state)
                        self.path.append(curr_i)

                    else:
                        self.update_connection(curr_i, attempted_next_state, connection=False,
                                               add_new_node=self.there_is_another_node(curr_i))
                        print('understood that ', curr_i, attempted_next_state, 'are not connected')
                        in_plan = False
        else:
            DEBUG = 1
            raise ValueError('curr_i is h')
            curr_i = shortest_path[0]
            if self.anim is not None:
                self.anim.event_source.stop()
        return curr_i, last_i

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

        row_to_duplicate = self.perceivedDistances.loc[name]
        self.perceivedDistances = pd.concat([self.perceivedDistances, row_to_duplicate.to_frame().T], axis=0)
        self.perceivedDistances.index = pd.Index(list(self.perceivedDistances.index[:-1]) + [new_name])

        column_to_duplicate = self.perceivedDistances.loc[name]
        # append to column_to_duplicate a row with index new_name
        column_to_duplicate = pd.concat([column_to_duplicate, pd.Series([np.inf], index=[new_name])], axis=0)

        self.perceivedDistances = pd.concat([self.perceivedDistances, column_to_duplicate.to_frame()], axis=1)
        self.perceivedDistances.columns = pd.Index(list(self.perceivedDistances.columns[:-1]) + [new_name])

    def run(self):
        global curr_i, last_i
        curr_i = self.initial_state
        last_i = self.initial_state
        self.G = self.create_graph()
        self.path = [curr_i]
        while curr_i != self.final_node:
            self.update_state()
            self.i += 1
            # if self.i  == 4:
            #     DEBUG = 1

    @staticmethod
    def save(paths):
        with open("states_small.json", 'w') as f:
            json.dump(paths, f)

    def animate(self): # plt.style.use('dark_background')
        fig, ax = plt.subplots()
        im = ax.imshow(img)

        # ax.plot(self.seeds[:, 0], self.seeds[:, 1], 'ro')
        points_plot = ax.plot(self.points['x'], self.points['y'], 'ko')[0]

        dude_plot = ax.plot([self.points.loc[self.initial_state, 'x']],
                            [self.points.loc[self.initial_state, 'y']],
                            'ro', markersize=10)[0]

        global curr_i, last_i, attempt_i
        curr_i = self.initial_state
        last_i = self.initial_state
        attempt_i = self.initial_state

        self.paths = [[curr_i]]
        plt.tight_layout()

        def other_update_anim(frame):
            self.update_state()
            dude_plot.set_data([self.points.loc[curr_i, 'x']],
                               [self.points.loc[curr_i, 'y']])
            # add a line to ax
            ax.plot([self.points.loc[last_i, 'x'], self.points.loc[curr_i, 'x']],
                    [self.points.loc[last_i, 'y'], self.points.loc[curr_i, 'y']], 'k-')

            # add a line to ax
            if attempt_i != curr_i:
                ax.plot([self.points.loc[attempt_i, 'x'], self.points.loc[curr_i, 'x']],
                        [self.points.loc[attempt_i, 'y'], self.points.loc[curr_i, 'y']], 'r-')

            return dude_plot

        self.anim = animation.FuncAnimation(fig, other_update_anim, interval=400, repeat=False, frames=range(40))
        plt.show()

    @staticmethod
    def distances(points):
        l = points.shape[0]
        points_array = np.array(points)
        deltas = np.repeat(points_array, l, 0).reshape((l, l, 3))
        deltas -= deltas.swapaxes(0, 1)
        dists = pd.DataFrame(np.linalg.norm(deltas, axis=-1), index=points.index, columns=points.index)
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
        save_fig(fig, name='human_simulation_bar_chart')

    @classmethod
    def df(cls):
        with open(home + "\\PS_Search_Algorithms\\human_network_simulation\\states_small.json", 'r') as f:
            paths = json.load(f)
        # create dataframe with colums: name,
        tuples = [(name, 'Small', True, path) for name, path in enumerate(paths)]
        df = pd.DataFrame(tuples, columns=['name', 'size', 'winner', 'states_series'])
        return df

    def update_connection(self, state1, state2, connection=False, add_new_node=False):
        if add_new_node:
            self.add_new_node(curr_i)
        self.perceivedDistances.loc[state1, state2] = np.inf
        self.perceivedDistances.loc[state2, state1] = np.inf
        self.G = self.create_graph()

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
    for i in tqdm(range(50)):
        # i = 62
        stateMachine = HumanStateMachine(seed=i)
        # TODO: get percentages of first decisions for every junction for single experiments
        # TODO: with the distances,
        # TODO: at every junction, get the probabilities that humans believe that this junction is open.

        # TODO: from the percentages of the single experiments at get the probabilities for first decision
        stateMachine.run()
        paths.append(stateMachine.path)
        stateMachine.merge()
        DEBUG = 1

    stateMachine.plot_bar_chart(paths)
    stateMachine.save(paths)
    # stateMachine.animate()
    # print(stateMachine.paths)

    # Can we capture more of the dynamics ... for example movement inside of the state?
    # TODO: plot the time spent in each state -> distribution
    # TODO: Are the long time spans just a lot of waiting?
    # TODO: if we wanted to capture waiting...
