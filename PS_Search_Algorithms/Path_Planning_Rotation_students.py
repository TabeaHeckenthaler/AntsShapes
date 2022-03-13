from copy import copy
import os
import numpy as np
from typing import Union
import networkx as nx
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from Setup.Maze import start, end
from PS_Search_Algorithms.Path_planning_in_CS import Path_planning_in_CS, Node3D, Node2D, Node_constructors
from ConfigSpace.ConfigSpace_Maze import ConfigSpace
import pandas as pd
from Directories import home
from matplotlib import pyplot as plt
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from itertools import product
import pickle
from Directories import PhaseSpaceDirectory
import time


class Binned_ConfigSpace(ConfigSpace):
    def __init__(self, high_resolution_space: ConfigSpace, resolution: int):
        self.high_resolution_space = high_resolution_space
        self.dim = high_resolution_space.space.ndim
        self.resolution = resolution
        super().__init__(space=self.decimate_space())
        self.dual_space = self.calc_dual_space(periodic=[False, False, True])
        # self.draw_dual_space()
        self.node_constructor = Node_constructors[self.dim]

    def directory(self):
        return os.path.join(PhaseSpaceDirectory, 'SPT_decimated' + '_' + str(self.resolution))

    def draw_bins(self):
        pass
        # mask = np.zeros(self.conf_space.space.shape)
        # mask[120: 120 + 40, 120: 120 + 20, 300: 300 + 20] = True
        # self.conf_space.visualize_space(space=mask, colormap='Oranges')

    def decimate_space(self):
        # if os.path.exists(self.directory()):
        #     decimated_space = pickle.load(open(self.directory(), 'rb'))
        #     return decimated_space

        bin_size = self.resolution
        space = self.high_resolution_space.space

        decimated_shape = [int(s / bin_size) for s in space.shape]
        decimated_space = np.empty(decimated_shape)

        for i, j, k in product(*[range(s) for s in decimated_shape]):
            bin_sum = np.sum(self.bin([(i, j, k), (i, j, k)]))
            decimated_space[i, j, k] = bin_sum / bin_size ** self.dim
        # self.save(decimated_space)
        return decimated_space

    def save(self, decimated_space):
        pickle.dump(decimated_space, open(self.directory(), 'wb'))

    def bin(self, bin_indices: list) -> np.array:
        bin_size = self.resolution
        dsi = bin_indices  # decimated space indices
        return self.high_resolution_space.space[int(dsi[0][0] * bin_size):int((dsi[1][0] + 1) * bin_size),
               int(dsi[0][1] * bin_size):int((dsi[1][1] + 1) * bin_size),
               int(dsi[0][2] * bin_size):int((dsi[1][2] + 1) * bin_size)]

    def bin_cut_out(self, indices: list) -> tuple:
        """
        Return only the cut out part where all the indices (in high resolution space) lie in.
        The indices have to be in adjacent bins. If indices are not in adjacent bins, raise an Error.
        :param indices: indices in self.high_resolution_space
        :return: tuple of position of top, left index of cut_out in original cs, and actual cut_out
        """
        bin_size = self.resolution
        dsi = np.array(list(list(int(np.floor(indices[i][j] / bin_size)) for j in range(self.dim)) for i in range(len(indices))))

        bins = self.high_resolution_space.space[np.min(dsi[:, 0]) * bin_size:(np.max(dsi[:, 0]) + 1) * bin_size,
                                                np.min(dsi[:, 1]) * bin_size:(np.max(dsi[:, 1]) + 1) * bin_size,
                                                np.min(dsi[:, 2]) * bin_size:(np.max(dsi[:, 2]) + 1) * bin_size]

        cutout_tuple = (tuple(np.min(dsi[:, i]) * bin_size for i in range(self.dim)), bins)
        return cutout_tuple

    def ind_in_bin(self, bin_index: tuple) -> list:

        """

        :param bin_index:
        :return: list with all indices (in self.high_resolutions) as tuples in a bin
        """
        """
        grid = [(0, 0), (1, 0), (0, 1), (1, 1)]
        top_left = [self.corner_of_bin_in_space(bin_index) for _ in range(len(grid))]

        space_indices = [tuple(sum(x) for x in zip(tuple1, tuple2)) for tuple1, tuple2 in zip(top_left, grid)]
        """

        dim = len(bin_index)

        bin_size = self.resolution

        bin_shape = tuple(bin_size for _ in range(dim))

        # bin_ind = self.corner_of_bin_in_space(bin_index)

        space_indices = []

        for i in np.ndindex(bin_shape):
            index = np.array(i) + np.array(bin_index) * bin_size

            space_indices.append(tuple(index))

        return space_indices

    def space_ind_to_bin_ind(self, space_ind: tuple) -> tuple:
        """
        Find the bin, which contains the space index
        :param space_ind: index of node in high_resolution_space
        :return: bin indices
        """
        return tuple(int(ind / self.resolution) for ind in space_ind)

    # def corner_of_bin_in_space(self, bin_ind: tuple) -> tuple:
    #     """
    #     Find space index of the top left node inside the bin.
    #     :param bin_ind: index of bin in self.space
    #     :return: space indices
    #     """
    #     return tuple(int(ind * self.resolution) for ind in bin_ind)

    def find_path(self, start: tuple, end: tuple) -> tuple:
        """

        :param start: indices of first node (in self.known_config_space.binned_space)
        :param end: indices of second node (in self.known_config_space.binned_space)
        :return: (boolean (whether a path was found), list (node indices that connect the two indices))
        """
        top_left, bins = self.bin_cut_out([start, end])
        start_indices_in_bin = tuple(np.array(start) - np.array(top_left))
        end_indices_in_bin = tuple(np.array(end) - np.array(top_left))

        # cs = ConfigSpace(space=bins)
        # if cs.dual_space is None:
        #     cs.dual_space = cs.calc_dual_space()
        # try:
        #     shortest_path = nx.shortest_path(cs.dual_space,
        #                                      tuple(np.array(start) - np.array(top_left)),
        #                                      tuple(np.array(end) - np.array(top_left)))
        # except nx.NetworkXNoPath:
        #     return False, None
        # return True, shortest_path

        Planner = Path_planning_in_CS(self.node_constructor(*start_indices_in_bin, ConfigSpace(bins)),
                                      self.node_constructor(*end_indices_in_bin, ConfigSpace(bins)),
                                      conf_space=ConfigSpace(bins), periodic=(0, 0, 0))
        Planner.path_planning()
        if Planner.winner:
            return Planner.winner, Planner.generate_path()
        else:
            return False, None


class Path_Planning_Rotation_students(Path_planning_in_CS):
    def __init__(self, conf_space, start: Union[Node2D, Node3D], end: Union[Node2D, Node3D], resolution: int,
                 max_iter: int = 100000, dil_radius: int = 0):
        super().__init__(start, end, max_iter, conf_space=conf_space)
        # self.dil_radius = dil_radius
        self.resolution = resolution
        self.planning_space.space = copy(self.conf_space.space)
        self.warp_planning_space()
        self.found_path = None  # saves paths, so no need to recalculate

    def step_to(self, greedy_node) -> None:
        """
        Walk to the greedy node.
        :param greedy_node: greedy node with indices from high_resolution space.
        """
        print('I am stepping to: ', greedy_node.ind())
        greedy_node.parent = copy([self.found_path])
        self.found_path = None
        self._current = greedy_node

    def warp_planning_space(self):
        """
        Planning_space is a low resolution representation of the real maze.
        """
        # if self.dil_radius > 0:
        #     self.planning_space.space = self.conf_space.dilate(self.planning_space.space, radius=self.dil_radius)
        self.planning_space = Binned_ConfigSpace(self.planning_space, self.resolution)

    def distances_in_surrounding_nodes(self) -> dict:
        connected_bins = {bin_ind: self.planning_space.ind_in_bin(bin_ind)
                          for bin_ind in self.current_known().connected(space=self.planning_space.space)}
        connected_distance = {}
        for bin_ind, space_ind_list in connected_bins.items():
            for space_ind in space_ind_list:
                connected_distance[space_ind] = self.distance[bin_ind]
        return connected_distance

    def find_greedy_node(self) -> Union[Node2D, Node3D]:
        """
        Find the node with the smallest distance from self.end, that is bordering the self._current in
        self.planning_space.space
        :return: greedy node with indices from self.conf_space.space
        """

        path = nx.dijkstra_path(self.planning_space.dual_space,
                                self.planning_space.space_ind_to_bin_ind(self._current.ind()),
                                self.planning_space.space_ind_to_bin_ind(self.end.ind()),
                                weight="weight")

        if len(path) < 2:
            return self.node_constructor(*self.end.ind(), self.conf_space)

        else:
            greedy_bin = path[1]
            print('choose bin: ', greedy_bin)
            # connected_distance = self.distances_in_surrounding_nodes()

            # while len(connected_distance) > 0:
            minimal_nodes = self.planning_space.ind_in_bin(greedy_bin)
            random_node_in_greedy_bin = minimal_nodes[np.random.choice(len(minimal_nodes))]
            print('choose node: ', random_node_in_greedy_bin)
            return self.node_constructor(*random_node_in_greedy_bin, self.conf_space)

        # # I think I added this, because they were sometimes stuck in positions impossible to exit.
        # if np.sum(np.logical_and(self._current.surrounding(random_node_in_greedy_bin), self.voxel)) > 0:
        #     return self.node_constructor(*random_node_in_greedy_bin, self.conf_space)
        # else:
        #     connected_distance.pop(random_node_in_greedy_bin)
        # raise Exception('Not able to find a path')

    def is_winner(self):
        if self._current.ind() == self.end.ind():
            self.winner = True
            return self.winner
        return self.winner

    def possible_step(self, greedy_node: Union[Node2D, Node3D]) -> Union[bool]:
        """
        Check if walking from self._current to greedy_node is possible.
        :param greedy_node: in self.planning_space.high_resolution
        :return:
        """
        path_exists, self.found_path = self.planning_space.find_path(self._current.ind(), greedy_node.ind())

        if path_exists:
            # bin_index = self.planning_space.space_ind_to_bin_ind(greedy_node.ind())
            return True
            # manage_to_pass = np.random.uniform(0, 1) < self.planning_space.space[bin_index]
            # if manage_to_pass:
            #     return True
            # else:
            #     return False
        else:
            return False

    # def draw_dual_lattice(self):  # the function which draws a lattice defined as networkx grid
    #
    #     lattice = self.planning_space.dual_space
    #
    #     plt.figure(figsize=(6, 6))
    #     pos = {(x, y): (y, -x) for x, y in lattice.nodes()}
    #     nx.draw(lattice, pos=pos,
    #             node_color='yellow',
    #             with_labels=True,
    #             node_size=600)
    #
    #     edge_labs = dict([((u, v), d["weight"]) for u, v, d in lattice.edges(data=True)])
    #
    #     nx.draw_networkx_edge_labels(lattice,
    #                                  pos,
    #                                  edge_labels = edge_labs)

    def add_knowledge(self, greedy_bin: Union[Node2D, Node3D]) -> None:
        """
        No path was found in greedy node, so we need to update our self.speed.
        Some kind of Bayesian estimation... not
        :param greedy_bin: node in high resolution config_space
        """

        # self.draw_dual_lattice(); plt.show()
        print('Recalculating...')
        start_bin_ind = self.planning_space.space_ind_to_bin_ind(self._current.ind())
        end_bin_ind = self.planning_space.space_ind_to_bin_ind(greedy_bin.ind())

        edge = (start_bin_ind, end_bin_ind)
        weight = self.planning_space.dual_space.edges[edge]["weight"]
        nx.set_edge_attributes(self.planning_space.dual_space, {edge: {"weight": 2 * weight}})

        # self.planning_space.draw_dual_space(); plt.show()

    def compute_distances(self) -> None:
        """
        Computes travel time ( = self.distance) of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s and 0s. From the 0 line the distance metric will e calculated.
        pass

        # phi = np.ones_like(self.planning_space.space, dtype=int)
        # phi[self.planning_space.space_ind_to_bin_ind(self.end.ind())] = 0
        # self.distance = travel_time(phi, self.speed, periodic=self.periodic)

    def current_known(self) -> Union[Node2D, Node3D]:
        """
        :return: bin index of where the self._current lies in.
        """
        return self.node_constructor(*self.planning_space.space_ind_to_bin_ind(self._current.ind()),
                                     ConfigSpace(self.planning_space.space))

    def draw_maze(self):
        self.conf_space.fig = plt.imshow(self.conf_space.space)
        plt.show(block=False)

    def generate_path(self, length=np.infty, ind=False) -> np.array:
        """
        Generates path from current node, its parent node, and parents parents node etc.
        Returns an numpy array with the x, y, and theta coordinates of the path,
        starting with the initial node and ending with the current node.
        :param length: maximum length of generated path
        :param ind:
        :return: np.array with [[x1, y1, angle1], [x2, y2, angle2], ... ] of the path
        """
        path = [self._current.coord()]
        node = self._current
        i = 0
        while node.parent is not None and i < length:
            if not ind:
                path.insert(0, node.parent.ind())
            else:
                path.append(node.parent.ind())
            node = node.parent
            i += 1
        return np.array(path)


# this is only for testing
directory = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
resolution = 2
# binned_space = pd.read_excel(io=directory, sheet_name='binned_space').to_numpy()

if __name__ == '__main__':
    # 2D
    # directory = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
    # resolution = 2
    # binned_space = pd.read_excel(io=directory, sheet_name='binned_space').to_numpy()
    # config_space = ConfigSpace(space=pd.read_excel(io=directory, sheet_name='space').to_numpy())
    # Planner = Path_Planning_Rotation_students(conf_space=config_space,
    #                                           start=Node2D(1, 1, config_space),
    #                                           end=Node2D(7, 5, config_space),
    #                                           resolution=resolution)
    #
    # # Planner.draw_maze()
    # Planner.path_planning(display_cs=False)
    # DEBUG = 1

    x = Trajectory_ps_simulation(size='Large', shape='SPT', solver='human',
                                 geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
    conf_space = ConfigSpace_Maze('human', 'Large', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
    conf_space.load_space()
    conf_space.visualize_space()

    # starting_indices = conf_space.coords_to_indices(*start(x, 'back'))
    # ending_indices = conf_space.coords_to_indices(*end(x))

    # starting_indices = (91, 126, 313)  # used to be (91, 126, 0)
    ending_indices = (336, 126, 313)
    stuck_indices = (215, 139, 352)
    # (123, 135, 301)

    Planner = Path_Planning_Rotation_students(conf_space=conf_space,
                                              start=Node3D(*stuck_indices, conf_space),
                                              end=Node3D(*ending_indices, conf_space),
                                              resolution=40)

    # Planner.draw_maze()
    Planner.path_planning(display_cs=True)
    DEBUG = 1

# cannot decimate high-resolution space exactly
# motivate your choice of initializing and updating weights
