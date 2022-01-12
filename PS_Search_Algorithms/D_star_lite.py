from PhaseSpaces import PhaseSpace, PS_transformations
from trajectory_inheritance.trajectory import Trajectory
from Directories import SaverDirectories
from Setup.Maze import start, end, Maze
from PS_Search_Algorithms.classes.Node_ind import Node_ind
from copy import copy
from mayavi import mlab
import os
import numpy as np
from Analysis.GeneralFunctions import graph_dir
from skfmm import travel_time  # use this! https://pythonhosted.org/scikit-fmm/
from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
from PS_Search_Algorithms.Dstar_functions import voxel
from scipy.ndimage.measurements import label
from Directories import ps_path

structure = np.ones((3, 3, 3), dtype=int)


class D_star_lite:
    r"""
    Class for path planning
    """

    def __init__(self,
                 starting_node,
                 ending_node,
                 conf_space,
                 known_conf_space,
                 max_iter=100000,
                 average_radius=None,
                 ):
        r"""
        Setting Parameter

        start:Start Position [x,y] (current node will be set to this)
        end_screen:Goal Position [x,y] (I only take into account the x coordinate in my mazes... its more like a finish line)
        conf_space:Configuration Space [PhaseSpace]
        known_conf_space:configuration space according to which the solver plans his path before taking his first step

        Keyword Arguments:
            * *max_inter* [int] --
              after how many iterations does the solver stop?
            * *average_radius* [int] --
              average radius of the load
        """
        self.max_iter = max_iter

        self.conf_space = conf_space  # 1, if there is a collision, otherwise 0
        self.known_conf_space = known_conf_space
        self.known_conf_space.initialize_maze_edges()
        self.distance = None
        self.average_radius = average_radius

        # this is just an example for a speed
        self.speed = np.ones_like(conf_space.space)
        self.speed[:, int(self.speed.shape[1] / 2):-1, :] = copy(self.speed[:, int(self.speed.shape[1] / 2):-1, :] / 2)

        # Set current node as the start node.
        self.start = Node_ind(*starting_node, self.conf_space.space.shape, average_radius)

        if self.collision(self.start):
            raise Exception('Your start is not in configuration space')
        self.end = Node_ind(*ending_node, self.conf_space.space.shape, average_radius)

        if self.collision(self.end):
            print('Your end is not in configuration space')
            self.end = Node_ind(*self.end.find_closest_possible_conf(conf_space), self.conf_space.space.shape, average_radius)

        self.current = self.start
        self.winner = False

    def planning(self, sensing_radius=7):
        r"""
        d star path planning
        While the current node is not the end_screen node, and we have iterated more than max_iter
        compute the distances to the end_screen node (of adjacent nodes).
        If distance to the end_screen node is inf, break the loop (there is no solution).
        If distance to end_screen node is finite, find node connected to the
        current node with the minimal distance (+cost) (next_node).
        If you are able to walk to next_node is, make next_node your current_node.
        Else, recompute your distances.


        :Keyword Arguments:
            * *sensing_radius* (``int``) --
              At an interception with the wall, sensing_radius gives the radius of the area of knowledge added to
              the solver around the point of interception.
        """
        # TODO: WAYS TO MAKE LESS EFFICIENT:
        #  limited memory
        #  locality (patch size)
        #  accuracy of greedy node, add stochastic behaviour
        #  false walls because of limited resolution

        self.compute_distances(self.known_conf_space)
        # _ = self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        # _ = self.draw_conf_space_and_path(self.known_conf_space, 'known_conf_space_fig')

        for ii, _ in enumerate(range(self.max_iter)):
            # if self.current.xi < self.end.xi:  # TODO: more general....
            if self.current.ind() != self.end.ind():
                if self.current.distance == np.inf:
                    return None  # cannot find path

                greedy_node = self.find_greedy_node(self.known_conf_space)
                if not self.collision(greedy_node):
                    greedy_node.parent = copy(self.current)
                    self.current = greedy_node

                else:
                    self.add_knowledge(greedy_node, sensing_radius=sensing_radius)
                    self.compute_distances(self.known_conf_space)
            else:
                self.winner = True
                return self
        return self

    def add_knowledge(self, central_node, sensing_radius=7):
        r"""
        Adds knowledge to the known configuration space of the solver with a certain sensing_radius around
        the central node, which is the point of interception
        """
        # roll the array
        rolling_indices = [- max(central_node.xi - sensing_radius, 0),
                           - max(central_node.yi - sensing_radius, 0),
                           - (central_node.thetai - sensing_radius)]

        conf_space_rolled = np.roll(self.conf_space.space, rolling_indices, axis=(0, 1, 2))
        known_conf_space_rolled = np.roll(self.known_conf_space.space, rolling_indices, axis=(0, 1, 2))

        # only the connected component which we sense
        sr = sensing_radius
        labeled, _ = label(conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], structure)
        known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr] = \
            np.logical_or(
                np.array(known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], dtype=bool),
                np.array(labeled == labeled[sr, sr, sr])).astype(int)

        # update_screen known_conf_space by using known_conf_space_rolled and rolling back
        self.known_conf_space.space = np.roll(known_conf_space_rolled, [-r for r in rolling_indices], axis=(0, 1, 2))

    def compute_distances(self, conf_space):
        r"""
        Computes distance of the current position of the solver to the finish line in conf_space
        """
        # phi should contain -1s and 1s, later from the 0 line the distance metric will be calculated.
        phi = np.ones_like(conf_space.space)

        # mask
        mask = conf_space.space is False
        # phi.data should contain -1s and 1s and phi.mask should contain a boolean array
        phi = np.ma.MaskedArray(phi, mask)

        # here the finish line is set to 0
        phi[self.end.xi, :, :] = 0

        # calculate the distances from the goal position
        # self.distance = distance(phi, periodic=(0, 0, 1)).data this is the easiest, if the speed is uniform

        # if the speed is not uniform:
        self.distance = travel_time(phi, self.speed, periodic=(0, 0, 1)).data

        # how to plot your results in 2D in a certain plane
        # plot_distances(self, index=self.current.yi, plane='y')
        # plot_distances(self, index=self.current.xi, plane='x')
        return

    def find_greedy_node(self, conf_space):
        """
        Find the node with the smallest distance from self.end, that is bordering the self.current.
        :param conf_space:
        :return:
        """
        connected = self.current.connected(conf_space)

        while True:
            list_distances = [self.distance[node_indices] for node_indices in connected]

            if len(list_distances) == 0:
                raise Exception('Not able to find a path')

            minimal_nodes = np.where(list_distances == np.array(list_distances).min())[0]
            greedy_one = np.random.choice(minimal_nodes)
            greedy_node_ind = connected[greedy_one]

            if np.sum(np.logical_and(~self.current.surrounding(conf_space, greedy_node_ind), voxel)) > 0:
                node = Node_ind(*greedy_node_ind, conf_space.space.shape, self.average_radius)
                return node
            else:
                connected.remove(greedy_node_ind)

    def collision(self, node):
        """
        finds the indices_to_coords of (x, y, theta) in conf_space,
        where angles go from (0 to 2pi)
        """
        return not self.conf_space.space[node.xi, node.yi, node.thetai]

    def into_trajectory(self, size='XL', shape='SPT', solver='ps_simulation', filename='Dlite'):
        path = self.generate_path()
        x = Trajectory(size=size,
                       shape=shape,
                       solver=solver,
                       filename=filename,
                       winner=True)

        x.position = path[:, :2]
        x.angle = path[:, 2]
        x.frames = np.array([i for i in range(x.position.shape[0])])
        return x

    def draw_conf_space_and_path(self, conf_space, fig_name):
        fig = conf_space.visualize_space(name=fig_name)
        self.start.draw_node(conf_space, fig=fig, scale_factor=0.5, color=(0, 0, 0))
        self.end.draw_node(conf_space, fig=fig, scale_factor=0.5, color=(0, 0, 0))

        path = self.generate_path()
        conf_space.draw(fig, path[:, 0:2], path[:, 2], scale_factor=0.2, color=(1, 0, 0))
        return fig

    def generate_path(self, length=np.infty, ind=False):
        r"""
        Generates path from current node, its parent node, and parents parents node etc.
        Returns an numpy array with the x, y, and theta coordinates of the path,
        starting with the initial node and ending with the current node.
        """
        path = [self.current.coord(self.conf_space)]
        node = self.current
        i = 0
        while node.parent is not None and i < length:
            if not ind:
                path.insert(0, node.parent.coord(self.conf_space))
            else:
                path.append(node.parent.ind())
            node = node.parent
            i += 1
        return np.array(path)

    def show_animation(self, save=False):
        conf_space_fig = self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        known_conf_space_fig = self.draw_conf_space_and_path(self.known_conf_space, 'known_conf_space_fig')
        if save:
            mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg',
                         magnification=4,
                         figure=conf_space_fig)
            mlab.savefig(graph_dir() + os.path.sep + self.conf_space.name + '.jpg',
                         magnification=4,
                         figure=known_conf_space_fig)
            # mlab.close()


def main(size='XL', shape='SPT', solver='ant', dil_radius=8, sensing_radius=7, show_animation=False, filename='test',
         save=False, starting_point=None, ending_point=None):
    print('Calculating: ' + filename)

    # ====Search Path with RRT====
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape)
    conf_space.load_space()

    if starting_point is None:
        starting_point = start(size, shape, solver)
    if ending_point is None:
        ending_point = end(size, shape, solver)

    # ====Set known_conf_space ====
    # 1) known_conf_space are just the maze walls
    # known_conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape + '_pp')
    # known_conf_space.load_space(path=ps_path(size, shape, solver, point_particle=True))

    # 2) dilated version of the conf_space

    known_conf_space = copy(conf_space)
    if dil_radius > 0:
        known_conf_space = known_conf_space.dilate(space=conf_space.space, radius=dil_radius)

    # ====Set Initial parameters====
    d_star_lite = D_star_lite(
        starting_node=conf_space.coords_to_indices(*starting_point),
        ending_node=conf_space.coords_to_indices(*ending_point),
        average_radius=Maze(size, shape, solver).average_radius(),
        conf_space=conf_space,
        known_conf_space=known_conf_space,
    )

    # ====Calculate the trajectory_inheritance the solver takes====
    d_star_lite_finished = d_star_lite.planning(sensing_radius=sensing_radius)
    path = d_star_lite_finished.generate_path()

    if not d_star_lite_finished.winner:
        print("Cannot find path")
    else:
        print("found path in {} iterations!!".format(len(path)))

    # === Draw final path ===
    if show_animation:
        d_star_lite_finished.show_animation(save=save)

    # ==== Turn this into trajectory_inheritance object ====
    x = d_star_lite_finished.into_trajectory(size=size, shape=shape, solver='ps_simulation', filename=filename)
    x.play(wait=200)
    if save:
        x.save()
        return
    return x


if __name__ == '__main__':
    size = 'S'
    solver = 'ant'


    def calc(sensing_radius, dil_radius, shape):
        filename = filename_dstar(size, shape, dil_radius, sensing_radius)

        if filename in os.listdir(SaverDirectories['ps_simulation']):
            pass
        else:
            main(size=size,
                 shape=shape,
                 solver=solver,
                 sensing_radius=sensing_radius,
                 dil_radius=dil_radius,
                 filename=filename,
                 starting_point=None,
                 ending_point=None,
                 )

    # === For parallel processing multiple trajectories on multiple cores of your computer ===
    # Parallel(n_jobs=6)(delayed(calc)(sensing_radius, dil_radius, shape)
    #                    for dil_radius, sensing_radius, shape in
    #                    itertools.product(range(0, 16, 1), range(1, 16, 1), ['SPT'])
    #                    # itertools.product([0], [0], ['H', 'I', 'T'])
    #                    )

    # === For processing a solver ===
    # calc(100, 0, 'SPT')
    calc(100, 0, 'T')
