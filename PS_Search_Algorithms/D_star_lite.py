from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
from Setup.Maze import start, end, Maze
from PS_Search_Algorithms.classes.Node_ind import Node_ind
from copy import copy
from mayavi import mlab
import os
import numpy as np
from Analysis.GeneralFunctions import graph_dir
from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
from scipy.ndimage.measurements import label
from PS_Search_Algorithms.Dstar_functions import voxel
from skfmm import travel_time, distance  # use this! https://pythonhosted.org/scikit-fmm/
from Directories import SaverDirectories

try:
    import cc3d
except:
    print('cc3d not installed')

structure = np.ones((3, 3, 3), dtype=int)


class D_star_lite:
    def __init__(self, x: Trajectory_ps_simulation, sensing_radius: int, dilation_radius: int, starting_point: tuple,
                 ending_point: tuple, max_iter: int = 100000) -> None:
        """
        Initialize the D_star_lite solver.
        :param x: Trajectory_ps_simulation object, that stores information on the maze dimensions.
        :param sensing_radius: radius around the point of impact, where the known_conf_space is updated by the real
        conf_space
        :param dilation_radius: radius by which the conf_space is dilated by to calculate known_conf_space.
        :param starting_point: coordinates of starting point (in cm or m)
        :param ending_point: coordinates of ending point (in cm or m)
        :param max_iter: maximal number of steps before the solver gives up and returns a trajectory with x.winner=False
        """

        # TODO: There is something wrong with passing x.solver...
        self.conf_space = PhaseSpace.PhaseSpace('human', x.size, x.shape, x.geometry())
        self.conf_space.load_space()
        # self.conf_space.visualize_space()

        known_conf_space = copy(self.conf_space)
        if dilation_radius > 0:
            known_conf_space = known_conf_space.dilate(space=self.conf_space.space, radius=dilation_radius)
        self.known_conf_space = known_conf_space
        self.known_conf_space.initialize_maze_edges()

        self.max_iter = max_iter
        self.sensing_radius = sensing_radius
        self.average_radius = Maze(x).average_radius()
        self.distance = None
        self.winner = False
        self.start, self.end = self.define_starting_and_ending(x, starting_point, ending_point)
        self.current = self.start

        # self.draw_conf_space_and_path()
        # self.speed = np.ones_like(self.conf_space.space) # just an example of a speed
        # self.speed[:, int(self.speed.shape[1] / 2):-1, :] =
        # copy(self.speed[:, int(self.speed.shape[1] / 2):-1, :] / 2)

    def define_starting_and_ending(self, x: Trajectory_ps_simulation, starting_point: tuple, ending_point: tuple) \
            -> tuple:
        """
        Define the starting and ending point of the solver.
        :param x: trajectory that carries information on the maze
        :param starting_point: coordinates of starting point (in cm or m)
        :param ending_point: coordinates of ending point (in cm or m)
        """
        if starting_point is None:
            starting_point = start(x)
        if ending_point is None:
            ending_point = end(x)
        start_ = Node_ind(*self.conf_space.coords_to_indices(*starting_point), self.conf_space, self.average_radius)
        end_ = Node_ind(*self.conf_space.coords_to_indices(*ending_point), self.conf_space, self.average_radius)

        if self.collision(start_):
            print('Your start is not in configuration space')
            # start_.draw_maze()
            # if bool(input('Move back? ')):
            if False:
                start_ = start_.find_closest_possible_conf(note='backward')
            else:
                start_ = start_.find_closest_possible_conf()

        if self.collision(end_):
            print('Your end is not in configuration space')
            # end_.draw_maze()
            # if bool(input('Move back? ')):
            if False:
                end_ = end_.find_closest_possible_conf(note='backward')
            else:
                end_ = end_.find_closest_possible_conf()
        return start_, end_

    def path_planning(self, display_cs=True) -> None:
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
        self.compute_distances()
        # _ = self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        # _ = self.draw_conf_space_and_path(self.known_conf_space, 'known_conf_space_fig')

        ii = 0
        while self.current.ind() != self.end.ind() and ii < self.max_iter:
            ii += 1
            if display_cs:
                self.current.draw_node(fig=self.conf_space.fig, scale_factor=0.2, color=(1, 0, 0))
            if self.current.distance == np.inf:
                return

            greedy_node = self.find_greedy_node()
            if not self.collision(greedy_node):
                greedy_node.parent = copy(self.current)
                self.current = greedy_node
            else:
                self.add_knowledge(greedy_node)
                self.compute_distances()

        if self.current.ind() == self.end.ind():
            self.winner = True

    def add_knowledge(self, central_node: Node_ind) -> None:
        """
        Adds knowledge to the known configuration space of the solver with a certain sensing_radius around
        the central node, which is the point of interception
        :param central_node: point of impact, which is the center of where the maze will be updated
        """
        # roll the array
        rolling_indices = [- max(central_node.xi - self.sensing_radius, 0),
                           - max(central_node.yi - self.sensing_radius, 0),
                           - (central_node.thetai - self.sensing_radius)]

        conf_space_rolled = np.roll(self.conf_space.space, rolling_indices, axis=(0, 1, 2))
        known_conf_space_rolled = np.roll(self.known_conf_space.space, rolling_indices, axis=(0, 1, 2))

        # only the connected component which we sense
        sr = self.sensing_radius
        labeled, _ = label(conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], structure)
        known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr] = \
            np.logical_or(
                np.array(known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], dtype=bool),
                np.array(labeled == labeled[sr, sr, sr])).astype(int)

        # update_screen known_conf_space by using known_conf_space_rolled and rolling back
        self.known_conf_space.space = np.roll(known_conf_space_rolled, [-r for r in rolling_indices], axis=(0, 1, 2))

    def unnecessary_space(self, buffer: int = 5):
        unnecessary = np.ones_like(self.conf_space.space, dtype=bool)
        unnecessary[np.min([self.start.ind()[0] - buffer, self.end.ind()[0] - buffer]):
                    np.max([self.start.ind()[0] + buffer, self.end.ind()[0] + buffer]),
        np.min([self.start.ind()[1] - buffer, self.end.ind()[1] - buffer]):
        np.max([self.start.ind()[1] + buffer, self.end.ind()[1] + buffer])] = False
        return unnecessary

    def compute_distances(self) -> None:
        """
        Computes distance of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s, later from the 0 line the distance metric will be calculated.
        phi = np.ones_like(self.known_conf_space.space, dtype=int)

        # mask
        mask = ~self.known_conf_space.space

        # this is to reduce computing power: we don't have to calculate distance in all space, just in small space
        # TODO: increase in a while loop
        # space = np.logical_and(~self.unnecessary_space(buffer=5), self.known_conf_space.space)
        # labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        # if labels[self.end.ind()] == labels[self.start.ind()]:
        #     mask = mask or self.unnecessary_space()
        #
        # else:
        #     space = np.logical_and(~self.unnecessary_space(buffer=50), self.known_conf_space.space)
        #     labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        #     if labels[self.end.ind()] == labels[self.start.ind()]:
        #         mask = np.logical_or(mask, self.unnecessary_space())

        # phi.data should contain -1s and 1s and zeros and phi.mask should contain a boolean array
        phi = np.ma.MaskedArray(phi, mask)
        phi.data[self.end.ind()] = 0

        # calculate the distances from the goal position, this is the easiest, if the speed is uniform
        print('Recompute distances')
        # self.distance = distance(phi, periodic=(0, 0, 1)).data
        # in order to mask the 'unreachable' nodes (diagonal or outside of conf_space), set distance there to inf.
        dist = distance(phi, periodic=(0, 0, 1))
        dist_data = dist.data
        dist_data[dist.mask] = np.inf
        self.distance = dist_data

        # if the speed is not uniform:
        # self.distance = travel_time(phi, self.speed, periodic=(0, 0, 1)).data

        # how to plot your results in 2D in a certain plane
        # self.conf_space.visualize_space()
        # self.conf_space.visualize_space(space=self.distance, colormap='Oranges')
        # plot_distances(self, index=self.current.xi, plane='x')

    def find_greedy_node(self) -> Node_ind:
        """
        Find the node with the smallest distance from self.end, that is bordering the self.current.
        """
        connected = self.current.connected()

        while True:
            list_distances = [self.distance[node_indices] for node_indices in connected]

            if len(list_distances) == 0:
                raise Exception('Not able to find a path')

            minimal_nodes = np.where(list_distances == np.array(list_distances).min())[0]
            greedy_one = np.random.choice(minimal_nodes)
            greedy_node_ind = connected[greedy_one]
            # loop: (115, 130, 383), (116, 130, 382), (117, 129, 381), (116, 131, 381)
            # return Node_ind(*greedy_node_ind, self.conf_space.space.shape, self.average_radius)

            # I think I added this, because they were sometimes stuck in positions impossible to exit.
            if np.sum(np.logical_and(self.current.surrounding(greedy_node_ind), voxel)) > 0:
                node = Node_ind(*greedy_node_ind, self.conf_space, self.average_radius)
                return node
            else:
                connected.remove(greedy_node_ind)

    def collision(self, node: Node_ind, space: np.array = None) -> bool:
        """
        :param space: space, which
        :param node: Node, which should be checked, whether in space.
        """
        if space is None:
            space = self.conf_space.space
        return not space[node.xi, node.yi, node.thetai]

    def generate_path(self, length=np.infty, ind=False) -> np.array:
        """
        Generates path from current node, its parent node, and parents parents node etc.
        Returns an numpy array with the x, y, and theta coordinates of the path,
        starting with the initial node and ending with the current node.
        :param length: maximum length of generated path
        :param ind:
        :return: np.array with [[x1, y1, angle1], [x2, y2, angle2], ... ] of the path
        """
        path = [self.current.coord()]
        node = self.current
        i = 0
        while node.parent is not None and i < length:
            if not ind:
                path.insert(0, node.parent.coord())
            else:
                path.append(node.parent.ind())
            node = node.parent
            i += 1
        return np.array(path)

    def into_trajectory(self, x: Trajectory_ps_simulation) -> Trajectory_ps_simulation:
        """
        Turn a D_star_lite object into the corresponding Trajectory object.
        :param x: Trajectory without position and angle etc.
        :return: Trajectory with position and angle
        """
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
        self.draw_conf_space_and_path(space=self.known_conf_space)
        if save:
            mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg', magnification=4)
            mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg', magnification=4)
            # mlab.close()


def run_dstar(shape: str, size: str, solver: str, dilation_radius: int = 8, sensing_radius: int = 7,
              filename: str = None, show_animation: bool = False, starting_point: tuple = None,
              ending_point: tuple = None, geometry: tuple = None) -> Trajectory_ps_simulation:
    """
    Initialize a trajectory, initialize a solver, run the path planning, pack it into a trajectory.
    :param shape: shape of the load, that moves through the maze
    :param size: size of the load
    :param solver: type of solver
    :param sensing_radius: radius around the point of impact, where the known_conf_space is updated by the real
    conf_space
    :param dilation_radius: radius by which the conf_space is dilated by to calculate known_conf_space.
    :param filename: Name of the trajectory
    :param show_animation: show animation
    :param starting_point: point (in real world coordinates) that the solver starts at
    :param ending_point: point (in real world coordinates) that the solver is aiming to reach
    :param geometry: geometry of the maze
    :return: trajectory object.
    """
    if filename is None:
        filename = filename_dstar(size, shape, dilation_radius, sensing_radius)
    # elif filename in os.listdir(SaverDirectories['ps_simulation']):
    #     return
    x = Trajectory_ps_simulation(size=size, shape=shape, solver=solver, filename=filename, geometry=geometry)
    d_star_lite = D_star_lite(x, sensing_radius=sensing_radius, dilation_radius=dilation_radius,
                              starting_point=starting_point, ending_point=ending_point)
    d_star_lite.path_planning()
    if show_animation:
        d_star_lite.show_animation()
    return d_star_lite.into_trajectory(x)


if __name__ == '__main__':
    x = run_dstar('SPT', 'Small Far', 'ps_simulation', dilation_radius=0, sensing_radius=100)
    x.play(wait=200)
    x.save()

    # TODO: WAYS TO MAKE LESS EFFICIENT:
    #  limited memory
    #  locality (patch size)
    #  accuracy of greedy node, add stochastic behaviour
    #  false walls because of limited resolution

    # === For parallel processing multiple trajectories on multiple cores of your computer ===
    # Parallel(n_jobs=6)(delayed(run_dstar)(sensing_radius, dil_radius, shape)
    #                    for dil_radius, sensing_radius, shape in
    #                    itertools.product(range(0, 16, 1), range(1, 16, 1), ['SPT'])
    #                    # itertools.product([0], [0], ['H', 'I', 'T'])
    #                    )

