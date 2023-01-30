from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from Setup.Maze import start, end, Maze
from PS_Search_Algorithms.classes.Node import Node3D, Node2D, Node_constructors
from copy import copy
import os
import numpy as np
from Analysis.GeneralFunctions import graph_dir
from PS_Search_Algorithms.classes.Node import voxel
from skfmm import distance  # use this! https://pythonhosted.org/scikit-fmm/
from typing import Union
from matplotlib import pyplot as plt
try:
    import cc3d
except:
    print('cc3d not installed')


class Path_planning_in_CS:
    """
    No diagonal paths allowed
    """
    def _get_distance(self):
        return self._distance

    def _set_distance(self, distance):
        if distance is not None and distance.shape != self.planning_space.space.shape:
            raise ValueError('Your distance does not have the right shape')
        self._distance = distance

    distance = property(_get_distance, _set_distance)

    def _get_current(self):
        return self._current

    def _set_current(self, next_current: Union[Node2D, Node3D]):
        if next_current not in list(self._current.iterate_surroundings()):
            raise ValueError('You are trying to set your current node to a far away node.')
        self._current = next_current

    current = property(_get_current, _set_current)

    def __init__(self, start: Union[Node3D, Node2D], end: Union[Node3D, Node2D], max_iter: int = 100000,
                 conf_space=None, periodic=(0, 0, 1)) -> None:
        self.start = start
        self.end = end
        self.max_iter = max_iter
        self.conf_space = conf_space
        self._current = self.start
        self.structure = np.ones(tuple((3 for _ in range(self.conf_space.space.ndim))), dtype=int)
        self.planning_space = conf_space
        # self.planning_space.space is the space according to which distances are calculated and nodes to walk to are
        # chosen, which are closest to the end (not necessarily in self.conf_space.space, if planning_space is warped.)
        self._distance = None
        self.winner = False
        self.voxel = voxel[self.conf_space.space.ndim]
        self.periodic = periodic[: self.conf_space.space.ndim]
        self.node_constructor = Node_constructors[self.conf_space.space.ndim]  # class of node (Node2D or Node3D)

    def is_winner(self):
        if self._current.ind() == self.end.ind():
            self.winner = True
            return self.winner
        return self.winner

    def path_planning(self, display_cs=False) -> None:
        """
        While the current node is not the end_screen node, and we have iterated more than max_iter
        compute the distances to the end_screen node (of adjacent nodes).
        If distance to the end_screen node is inf, break the loop (there is no solution).
        If distance to end_screen node is finite, find node connected to the
        current node with the minimal distance (+cost) (next_node).
        If you are able to walk to next_node is, make next_node your current_node.
        Else, recompute your distances.
        :param display_cs: Whether the path should be displayed during run time.
        """
        # print('Planning the path')
        self.compute_distances()
        # self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        # self.draw_conf_space_and_path(self.planning_space, 'planning_space_fig')

        ii = 0
        while ii < self.max_iter:
            ii += 1
            if self.is_winner():
                return
            if display_cs:
                self.conf_space.draw_ind(self._current.ind(), color=(1, 0, 0))
                # self._current.draw_node(fig=self.conf_space.fig,
                #                         # scale_factor=0.2, color=(1, 0, 0)
                #                         )

            if self.distance is not None and self.distance[self.current.ind()] == np.inf:
                return

            greedy_node = self.find_greedy_node()
            if self.possible_step(greedy_node):
                self.step_to(greedy_node)
            else:
                self.conf_space.draw_ind(greedy_node.ind(), color=(0.3, 0.3, 0.3), scale_factor=0.8)
                self.add_knowledge(greedy_node)
                self.compute_distances()

    def add_knowledge(self, *args):
        pass

    def collision(self, node: Union[Node2D, Node3D], space: np.array = None) -> bool:
        """
        :param space: space, which
        :param node: Node, which should be checked, whether in space.
        """
        if space is None:
            space = self.conf_space.space
        return not space[node.ind()]

    def possible_step(self, greedy_node) -> bool:
        return not self.collision(greedy_node)

    def find_greedy_node(self):
        """
        Find the node with the smallest distance from self.end, that is bordering the self._current in
        self.planning_space.space
        :return: greedy node with indices from self.conf_space.space
        """

        connected_nodes = self._current.connected(space=self.planning_space.space)
        connected_distance = {node: self.distance[node] for node in connected_nodes}

        while True:
            if len(connected_distance) == 0:
                raise Exception('Not able to find a path')

            minimal_nodes = list(filter(lambda x: connected_distance[x] == min(connected_distance.values()),
                                        connected_distance))
            random_minimal_node = minimal_nodes[np.random.choice(len(minimal_nodes))]

            # I think I added this, because they were sometimes stuck in positions impossible to exit.
            if np.sum(np.logical_and(self._current.surrounding(random_minimal_node), self.voxel)) > 0:
                return self.node_constructor(*random_minimal_node, self.conf_space)
            else:
                connected_distance.pop(random_minimal_node)

                # TODO: Tabea, choosing node, and then you already pop here?
                #  Then you have to update already here, right?

    def step_to(self, greedy_node) -> None:
        greedy_node.parent = copy(self._current)
        self._current = greedy_node

    def compute_distances(self) -> None:
        """
        Computes distance of the current position of the solver to the finish line according to self.planning_space.space
        """
        zero_contour_space = np.ones_like(self.planning_space.space, dtype=int)
        mask = ~np.array(self.planning_space.space, dtype=bool)
        zero_contour_space = np.ma.MaskedArray(zero_contour_space, mask)
        zero_contour_space[self.end.ind()] = 0

        # idea: increase in a while loop
        # this is to reduce computing power: we don't have to calculate distance in all space, just in small space
        # space = np.logical_and(~self.unnecessary_space(buffer=5), self.planning_space.space)
        # labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        # if labels[self.end.ind()] == labels[self.start.ind()]:
        #     mask = mask or self.unnecessary_space()
        #
        # else:
        #     space = np.logical_and(~self.unnecessary_space(buffer=50), self.planning_space.space)
        #     labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        #     if labels[self.end.ind()] == labels[self.start.ind()]:
        #         mask = np.logical_or(mask, self.unnecessary_space())

        # calculate the distances from the goal position, this is the easiest, if the speed is uniform
        # self.distance = distance(phi, periodic=(0, 0, 1)).data

        dist = distance(zero_contour_space, periodic=self.periodic)
        # in order to mask the 'unreachable' nodes (diagonal or outside of conf_space), set distance there to inf.
        dist.data[dist.mask] = np.inf
        self.distance = dist.data

        # if the speed is not uniform:
        # self.distance = travel_time(phi, self.speed, periodic=(0, 0, 1)).data

        # how to plot your results in 2D in a certain plane
        # self.conf_space.visualize_space()
        # self.conf_space.visualize_space(space=self.distance, colormap='Oranges')
        # plot_distances(self, index=self.current.xi, plane='x')

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
                path.insert(0, node.parent.coord())
            else:
                path.append(node.parent.ind())
            node = node.parent
            i += 1
        return np.array(path)


class Path_planning_in_Maze(Path_planning_in_CS):
    def __init__(self, x: Trajectory_ps_simulation, starting_node: Node3D, ending_node: Node3D, initial_cond: str,
                 max_iter: int = 100000) -> None:
        """
        Initialize the D_star_lite solver.
        :param x: Trajectory_ps_simulation object, that stores information on the maze dimensions.
        :param starting_point: coordinates of starting point (in cm or m)
        :param ending_point: coordinates of ending point (in cm or m)
        :param max_iter: maximal number of steps before the solver gives up and returns a trajectory with x.winner=False
        """
        conf_space = ConfigSpace_Maze(x.solver, x.size, x.shape, x.geometry())
        conf_space.load_space()
        # conf_space.visualize_space()

        self.average_radius = Maze(x).average_radius()
        super().__init__(starting_node, ending_node, max_iter, conf_space=conf_space)
        self.start, self.end = self.check_starting_and_ending(x, initial_cond=initial_cond)
        self._current = self.start
        # self.conf_space.visualize_space()

        self.planning_space = self.warp_conf_space()
        # self.planning_space.initialize_maze_edges()

        # self.draw_conf_space_and_path()
        # self.speed = np.ones_like(self.conf_space.space) # just an example of a speed
        # self.speed[:, int(self.speed.shape[1] / 2):-1, :] =
        # copy(self.speed[:, int(self.speed.shape[1] / 2):-1, :] / 2)

    def warp_conf_space(self) -> np.array:
        pass

    def check_starting_and_ending(self, x: Trajectory_ps_simulation, initial_cond: str) \
            -> tuple:
        """
        Define the starting and ending point of the solver.
        :param initial_cond: front or back of the maze
        :param x: trajectory that carries information on the maze
        """
        # start = None
        # end = None
        if self.start is None:
            starting_indices = self.conf_space.coords_to_indices(*start(x, initial_cond))
            self.start = Node3D(*starting_indices, self.conf_space)
        if self.end is None:
            ending_indices = self.conf_space.coords_to_indices(*end(x))
            self.end = Node3D(*ending_indices, self.conf_space)

        if self.collision(self.start):
            print('Your start is not in configuration space')
            # start_.draw_maze()
            # if bool(input('Move back? ')):
            if False:
                self.start = Node3D(*self.start.find_closest_possible_conf(note='backward'), self.conf_space)
            else:
                self.start = Node3D(*self.start.find_closest_possible_conf(), self.conf_space)

        if self.collision(self.end):
            print('Your end is not in configuration space')
            # end_.draw_maze()
            # if bool(input('Move back? ')):
            if False:
                self.end = Node3D(*self.end.find_closest_possible_conf(note='backward'), self.conf_space)
            else:
                self.end = Node3D(*self.end.find_closest_possible_conf(), self.conf_space)
        return self.start, self.end

    def unnecessary_space(self, buffer: int = 5):
        unnecessary = np.ones_like(self.conf_space.space, dtype=bool)
        unnecessary[np.min([self.start.ind()[0] - buffer, self.end.ind()[0] - buffer]):
                    np.max([self.start.ind()[0] + buffer, self.end.ind()[0] + buffer]),
        np.min([self.start.ind()[1] - buffer, self.end.ind()[1] - buffer]):
        np.max([self.start.ind()[1] + buffer, self.end.ind()[1] + buffer])] = False
        return unnecessary

    def into_trajectory(self, x: Trajectory_ps_simulation) -> Trajectory_ps_simulation:
        """
        Turn a D_star_lite object into the corresponding Trajectory object.
        :param x: Trajectory without position and angle etc.
        :return: Trajectory with position and angle
        """
        print('Putting path into trajectory object')
        path = self.generate_path()
        if not self.winner:
            print("Cannot find path")
        else:
            print("found path in {} iterations!!".format(len(path)))
        x.position = path[:, :2]
        x.angle = path[:, 2]
        x.frames = np.array([i for i in range(x.position.shape[0])])
        x.winner = self.winner
        return x

    def draw_conf_space_and_path(self, space=None) -> None:
        """
        Draw the configuration space and the path.
        :param space: space to draw
        """
        self.conf_space.visualize_space(space=space)
        self.start.draw_node(self.conf_space, fig=self.conf_space.fig, scale_factor=0.5, color=(0, 0, 0))
        self.end.draw_node(self.conf_space, fig=self.conf_space.fig, scale_factor=0.5, color=(0, 0, 0))

        path = self.generate_path()
        self.conf_space.draw(path[:, 0:2], path[:, 2], scale_factor=0.2, color=(1, 0, 0))

    def show_animation(self, save=False) -> None:
        """
        Show an animation of the solver
        :param save: Whether to save the image
        """
        self.draw_conf_space_and_path()
        self.draw_conf_space_and_path(space=self.planning_space)
        # if save:
        #     mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg', magnification=4)
        #     mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg', magnification=4)
            # mlab.close()
