from PhaseSpaces import PhaseSpace, PS_transformations
from trajectory import SaverDirectories, Trajectory, Save
from Setup.Load import average_radius
from Setup.Maze import start, end
from PS_Search_Algorithms.classes.Node_ind import Node_ind
from copy import copy
from progressbar import progressbar
from mayavi import mlab
import os
import numpy as np
from Analysis_Functions.GeneralFunctions import graph_dir
from skfmm import distance, travel_time # use this! https://pythonhosted.org/scikit-fmm/
import itertools
from joblib import Parallel, delayed
from Classes_Experiment.mr_dstar import filename_dstar
from PS_Search_Algorithms.Dstar_functions import voxel
from scipy.ndimage.measurements import label

structure = np.ones((3, 3, 3), dtype=int)


class D_star_lite:
    """
    Class for path planning
    """

    def __init__(self,
                 starting_point,
                 ending_point,
                 conf_space,
                 known_conf_space,
                 max_iter=100000,
                 av_radius=None,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        end:Goal Position [x,y]
        conf_space:Configuration Space [[x,y,size],...]

        Set current node as the start node.
        """
        self.max_iter = max_iter

        self.conf_space = conf_space  # 1, if there is a collision, otherwise 0
        self.known_conf_space = known_conf_space
        self.known_conf_space.initialize_maze_edges()

        self.average_radius = av_radius
        self.distance = None

        self.start = Node_ind(*starting_point, self.conf_space.space.shape, av_radius)
        if self.collision(self.start):
            raise Exception('Your start is not in configuration space')
        self.end = Node_ind(*ending_point, self.conf_space.space.shape, av_radius)
        if self.collision(self.end):
            raise Exception('Your end is not in configuration space')

        self.current = self.start
        self.winner = False

    def planning(self, sensing_radius=7):
        """
        d star path planning
        While the current node is not the end node, and we have iterated more than max_iter
        compute the distances to the end node (of adjacent nodes).
        If distance to the end node is inf, break the loop (there is no solution).
        If distance to end node is finite, find node connected to the
        current node with the minimal distance (+cost) (next_node).
        If you are able to walk to next_node is, make next_node your current_node.
        Else, recompute your distances.
        """
        # TODO: ways to make less efficient:
        # limited memory
        # locality (patch size)
        # accuracy of greedy node, add stochasticity
        # false walls because of limited resolution

        self.compute_distances(self.known_conf_space)
        # _ = self.draw_conf_space_and_path(self.conf_space, 'conf_space_fig')
        # _ = self.draw_conf_space_and_path(self.known_conf_space, 'known_conf_space_fig')

        for ii, _ in progressbar(enumerate(range(self.max_iter))):
            if self.current.xi < self.end.xi:
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
        # roll the array
        conf_space_rolled = np.roll(self.conf_space.space, [- max(central_node.xi - sensing_radius, 0),
                                                            - max(central_node.yi - sensing_radius, 0),
                                                            - (central_node.thetai - sensing_radius)],
                                    axis=(0, 1, 2))

        known_conf_space_rolled = np.roll(self.known_conf_space.space, [- max(central_node.xi - sensing_radius, 0),
                                                                        - max(central_node.yi - sensing_radius, 0),
                                                                        - (central_node.thetai - sensing_radius)],
                                          axis=(0, 1, 2))

        # only the connected component which we sense
        sr = sensing_radius
        labeled, _ = label(conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], structure)
        known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr] = \
            np.logical_or(
                np.array(known_conf_space_rolled[:2 * sr, :2 * sr, :2 * sr], dtype=bool),
                np.array(labeled == labeled[sr, sr, sr])).astype(int)

        # roll back
        self.known_conf_space.space = np.roll(known_conf_space_rolled, [max(central_node.xi - sensing_radius, 0),
                                                                        max(central_node.yi - sensing_radius, 0),
                                                                        (central_node.thetai - sensing_radius)],
                                              axis=(0, 1, 2))

    def compute_distances(self, conf_space):
        """

        """
        # phi should contain -1s and 1s, later from the 0 line the distance metric will be calculated.
        phi = np.ones_like(conf_space.space)

        # mask
        mask = conf_space.space == 1
        # phi.data should contain -1s and 1s and phi.mask should contain a boolean array
        phi = np.ma.MaskedArray(phi, mask)

        # here the finish line is set to 0
        phi[self.end.xi, :, :] = 0

        # calculate the distances from the goal position
        self.distance = distance(phi, periodic=(0, 0, 1)).data
        # self.plot_distances(index=current_indices[1], plane='y')
        return

    def find_greedy_node(self, conf_space):
        connected = self.current.connected(conf_space)
        while True:
            list_distances = [self.distance[node_indices] for node_indices in connected]

            if len(list_distances) == 0:
                raise Exception('Not able to find a path')

            greedy = np.where(list_distances == np.array(list_distances).min())[0][0]
            # greedy = np.random.choice(np.where(list_distances == np.array(list_distances).min())[0])
            greedy_node_ind = connected[greedy]

            if np.sum(np.logical_and(~self.current.surrounding(conf_space, greedy_node_ind), voxel)) > 0:
                node = Node_ind(*greedy_node_ind, conf_space.space.shape, self.average_radius)
                return node
            else:
                connected.remove(greedy_node_ind)

    def collision(self, node):
        """
        finds the indices of (x, y, theta) in conf_space,
        where angles go from (0 to 2pi)
        """
        if self.conf_space.space[node.xi, node.yi, node.thetai]:  # if there is a 1, we collide
            return True  # collision
        return False  # no collision

    def into_trajectory(self, size='XL', shape='SPT', solver='dstar', filename='Dlite'):
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
        fig = conf_space.visualize_space(fig_name)
        self.start.draw_node(conf_space, fig=fig, scale_factor=0.5, color=(0, 0, 0))
        self.end.draw_node(conf_space, fig=fig, scale_factor=0.5, color=(0, 0, 0))

        path = self.generate_path()
        conf_space.draw_trajectory(fig, path[:, 0:2], path[:, 2], scale_factor=0.2, color=(1, 0, 0))
        return fig

    def generate_path(self, length=np.infty, ind=False):
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


def main(size='XL', shape='SPT', solver='ant', dil_radius=8, sensing_radius=7, show_animation=False,
         filename='test', save=False):
    print('Calculating: ' + filename)

    # ====Search Path with RRT====
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape,
                                       name=size + '_' + shape)

    path = os.path.join(os.path.dirname(SaverDirectories[solver]),
                        PhaseSpace.ps_dir, solver, conf_space.name + ".pkl")
    conf_space.load_space(path=path)

    # known_conf_space = PhaseSpace.PhaseSpace(solver, size, shape,
    #                                          name=size + '_' + shape + '_pp')
    # known_conf_space.load_space(path=os.path.join(os.path.dirname(SaverDirectories[solver]),
    #                                               PhaseSpace.data_dir, solver, known_conf_space.name + ".pkl"))

    known_conf_space = copy(conf_space)

    if dil_radius > 0:
        known_conf_space = PS_transformations.dilation(known_conf_space, radius=dil_radius)

    # ====Set Initial parameters====
    d_star_lite = D_star_lite(
        starting_point=conf_space.coords_to_indexes(*start(size, shape, solver)),
        ending_point=conf_space.coords_to_indexes(*end(size, shape, solver)),
        av_radius=average_radius(size, shape, solver),
        conf_space=conf_space,
        known_conf_space=known_conf_space,
    )

    d_star_lite_finished = d_star_lite.planning(sensing_radius=sensing_radius)
    path = d_star_lite_finished.generate_path()

    if not d_star_lite_finished.winner:
        print("Cannot find path")
    else:
        print("found path in {} iterations!!".format(len(path)))

    # Draw final path
    if show_animation:
        d_star_lite_finished.show_animation(save=False)

    x = d_star_lite_finished.into_trajectory(size=size, shape=shape, solver='dstar', filename=filename)
    x.play(1, 'Display', wait=200)
    if save:
        Save(x)
    return


if __name__ == '__main__':
    size = 'XL'
    solver = 'ant'


    def calc(sensing_radius, dil_radius, shape):
        filename = filename_dstar(size, shape, dil_radius, sensing_radius)
        if filename in os.listdir(SaverDirectories['dstar']):
            pass
        else:
            main(size=size,
                 shape=shape,
                 solver=solver,
                 sensing_radius=sensing_radius,
                 dil_radius=dil_radius,
                 filename=filename
                 )
        #
        # main(size=size,
        #      shape=shape,
        #      solver=solver,
        #      sensing_radius=sensing_radius,
        #      dil_radius=dil_radius,
        #      filename=filename
        #      )


    # Parallel(n_jobs=6)(delayed(calc)(sensing_radius, dil_radius, shape)
    #                    for dil_radius, sensing_radius, shape in
    #                    itertools.product(range(0, 16, 1), range(1, 16, 1), ['SPT'])
    #                    # itertools.product([0], [0], ['H', 'I', 'T'])
    #                    )

    calc(100, 0, 'H')
