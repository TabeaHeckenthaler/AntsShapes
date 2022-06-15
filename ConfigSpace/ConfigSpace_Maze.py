import numpy as np
import pickle
from PhysicsEngine.Contact import possible_configuration
import os
import itertools
from Setup.Maze import Maze
from Directories import PhaseSpaceDirectory
from Analysis.resolution import resolution
from scipy import ndimage
from datetime import datetime
import string
from skfmm import distance
from tqdm import tqdm
from Analysis.PathPy.SPT_states import forbidden_transition_attempts, allowed_transition_attempts, states, short_forbidden
import networkx as nx
from matplotlib import pyplot as plt
from copy import copy
from Analysis.PathPy.SPT_states import cc_to_keep
from Analysis.GeneralFunctions import flatten
from Setup.Load import loops
from PIL import Image, ImageDraw

try:
    from mayavi import mlab
except:
    print('mayavi not installed')

try:
    import cc3d
except:
    print('cc3d not installed')

traj_color = (1.0, 0.0, 0.0)
start_end_color = (0.0, 0.0, 0.0)
scale = 5


# TODO: fix the x.winner attribute
# I want the resolution (in cm) for x and y and archlength to be all the same.


class ConfigSpace(object):
    def __init__(self, space: np.array, name=''):
        self.space = space  # True, if configuration is possible; False, if there is a collision with the wall
        self.name = name
        self.dual_space = None

    @staticmethod
    def reduced_resolution(space: np.array, reduction: int) -> np.array:
        """
        Reduce the resolution of the PS.
        :param space: space that you want to reshape
        :param reduction: By what factor the PS will be reduced.
        :return: np.array representing the CS with reduced resolution.
        """
        for axis in range(int(space.ndim)):
            space = space.take(indices=range(0, int(space.shape[axis] / reduction) * reduction), axis=axis)

        def reshape(array) -> np.array:
            """
            Shrink an array
            :return:
            """
            reshaper = [item for t in [(int(axis_len / reduction), reduction)
                                       for axis_len in array.shape] for item in t]

            return array.reshape(*reshaper)

        def summer(array) -> np.array:
            for i in range(int(array.ndim / 2)):
                array = array.sum(axis=i + 1)
            return array

        # return np.array(summer(reshape(space))/(reduction**space.ndim)>0.5, dtype=bool)
        return summer(reshape(space)) / (reduction ** space.ndim)

    def overlapping(self, ps_area):
        return np.any(self.space[ps_area.space])

    def draw_dual_space(self):  # the function which draws a lattice defined as networkx grid

        lattice = self.dual_space

        plt.figure(figsize=(6, 6))
        pos = {(x, y): (y, -x) for x, y in lattice.nodes()}
        nx.draw(lattice, pos=pos,
                node_color='yellow',
                with_labels=True,
                node_size=600)

        edge_labs = dict([((u, v), d["weight"]) for u, v, d in lattice.edges(data=True)])

        nx.draw_networkx_edge_labels(lattice,
                                     pos,
                                     edge_labels=edge_labs)
        plt.show()

    def neighbors(self, node) -> list:
        cube = list(np.ndindex((3, 3, 3)))
        cube.remove((1, 1, 1))
        neighbour_coords = np.array(cube) - 1
        a = neighbour_coords + np.array(node)
        a[:, 2] = a[:, 2] % self.space.shape[2]
        out_of_boundary = np.where(np.logical_or(np.logical_or(a[:, 0] >= np.array(self.space.shape)[0],
                                                               a[:, 1] >= np.array(self.space.shape)[1]),
                                                 np.logical_or(a[:, 0] < 0, a[:, 1] < 0)))[0]
        a1 = list(map(tuple, a.reshape((26, 3))))
        b = [tu for i, tu in enumerate(a1) if i not in out_of_boundary]
        return b

    def calc_dual_space(self, periodic=False) -> nx.grid_graph:
        dual_space = nx.grid_graph(dim=self.space.shape[::-1], periodic=periodic)

        nodes = list(copy(dual_space.nodes))

        for node in nodes:
            for neighbor in self.neighbors(node):
                dual_space.add_edge(node, neighbor)

        for edge in list(dual_space.edges):
            m = self.space[edge[0]] * self.space[edge[1]]
            if m == 0:
                dual_space.remove_edge(edge[0], edge[1])
            else:
                nx.set_edge_attributes(dual_space, {edge: {"weight": 1 - m}})
        return dual_space


class ConfigSpace_Maze(ConfigSpace):
    def __init__(self, solver: str, size: str, shape: str, geometry: tuple, name="", space=None):
        """

        :param solver: type of solver (ps_simluation, ant, human, etc.)
        :param size: size of the maze (XL, L, M, S)
        :param shape: shape of the load in the maze (SPT, T, H ...)
        :param geometry: tuple with names of the .xlsx files that contain the relevant dimensions
        :param name: name of the PhaseSpace.
        """
        super().__init__(space)
        maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)

        if len(name) == 0:
            name = size + '_' + shape

        self.name = name
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry

        x_range = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
        y_range = (0, maze.arena_height)

        self.extent = {'x': x_range,
                       'y': y_range,
                       'theta': (0, np.pi * 2)}
        self.average_radius = maze.average_radius()

        self.pos_resolution = self.extent['y'][1] / self.number_of_points()['y']
        self.theta_resolution = 2 * np.pi / self.number_of_points()['theta']

        self.space_boundary = None
        self.fig = None

        load = maze.bodies[-1]
        maze_corners = np.array_split(maze.corners(), maze.corners().shape[0]//4)
        load_corners = np.array(flatten(loops(load)))
        # loop_indices = [0, 1, 2, 3, 0]

        rect_edge_indices = np.array(((0, 1), (1, 2), (2, 3), (3, 0)))

        self.load_points = []
        self.load_edges = []
        for i, load_vertices_list in enumerate(np.array_split(load_corners, int(load_corners.shape[0]/4))):
            self.load_points.extend(load_vertices_list)
            self.load_edges.extend(rect_edge_indices + 4*i)
        self.load_points = np.array(self.load_points, float)

        self.maze_points = []
        self.maze_edges = []
        for i, maze_vertices_list in enumerate(maze_corners):
            self.maze_points.extend(maze_vertices_list)
            self.maze_edges.extend(rect_edge_indices + 4*i)

        # self.monitor = {'left': 3800, 'top': 160, 'width': 800, 'height': 800}
        # self.VideoWriter = cv2.VideoWriter('mayavi_Capture.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 20,
        #                                    (self.monitor['width'], self.monitor['height']))
        # self._initialize_maze_edges()

    @staticmethod
    def shift_by_pi(space):
        middle = space.shape[2]//2
        space = np.concatenate([space[:, :, middle:], space[:, :, :middle], ], axis=-1)
        return space

    def directory(self, point_particle: bool = False, erosion_radius: int = None, addition: str = '',
                  small: bool = False) \
            -> str:
        """
        Where a PhaseSpace should be saved, or where it can be found.
        :param erosion_radius: If the PhaseSpace is eroded, this should be added in the filename
        :param point_particle: not implemented
        :param addition: If the PhaseSpace already is calculated, it should be saved with a different filename, to not
        overwrite the old one.
        :param small: if you want only the labeled configuration space, not all the additional information
        :return: string with the name of the directory, where the PhaseSpace should be saved.
        """
        if self.size in ['Small Far', 'Small Near']:  # both have same dimensions
            filename = 'Small' + '_' + self.shape + '_' + self.geometry[0][:-5]
        else:
            filename = self.size + '_' + self.shape + '_' + self.geometry[0][:-5]

        if point_particle:
            return os.path.join(PhaseSpaceDirectory, self.shape, filename + addition + '_pp.pkl')
        if erosion_radius is not None:
            path_ = os.path.join(PhaseSpaceDirectory, self.shape, filename + '_labeled_erosion_'
                                 + str(erosion_radius) + addition + '.pkl')
            if small:
                path_ = path_[:-4] + '_small' + '.pkl'
            return path_

        path_ = os.path.join(PhaseSpaceDirectory, self.shape, filename + addition + '.pkl')
        if small:
            path_ = path_[:-4] + '_small' + '.pkl'
        return path_

    def number_of_points(self) -> dict:
        """
        How to pixelize the PhaseSpace. How many pixels along every axis.
        :return: dictionary with integers for every axis.
        """
        # x_num = np.ceil(self.extent['x'][1]/resolution)
        res = resolution(self.geometry, self.size, self.solver, self.shape)
        y_num = np.ceil(self.extent['y'][1] / res)
        theta_num = np.ceil(self.extent['theta'][1] * self.average_radius / res)
        return {'x': None, 'y': y_num, 'theta': theta_num}

    @staticmethod
    def calculate_distance(zero_distance_space: np.array, available_states: np.array) -> np.array:
        """
        Calculate the distance of every node in mask to space.
        :param zero_distance_space: Area, which has zero distance.
        :param available_states: Nodes, where distance to the zero_distance_space should be calculated.
        :return: np.array with distances from each node
        """

        # self.distance = distance(np.array((~np.array(self.space, dtype=bool)), dtype=int), periodic=(0, 0, 1))
        phi = np.array(~zero_distance_space, dtype=int)
        masked_phi = np.ma.MaskedArray(phi, mask=~available_states)
        d = distance(masked_phi, periodic=(0, 0, 1))
        return d

        # node at (105, 36, 102)

        # point = (105, 36, 102)
        # self.draw(self.indices_to_coords(*point)[:2], self.indices_to_coords(*point)[-1])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[point[0], :, :])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[:, point[1], :])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[:, :, point[2]])

    def initialize_maze_edges(self) -> None:
        """
        set x&y edges to 0 in order to define the maze boundaries (helps the visualization)
        :return:
        """
        self.space[0, :, :] = False
        self.space[-1, :, :] = False
        self.space[:, 0, :] = False
        self.space[:, -1, :] = False

    def calculate_space(self, mask=None) -> None:
        """
        This module calculated a space.
        :param point_particle:
        :param mask: If a mask is given only the unmasked area is calcuted. This is used to finetune maze dimensions.
        :return:
        This was beautifully created by Rotem!!!
        """
        maze = Maze(size=self.size, shape=self.shape, solver=self.solver, geometry=self.geometry)
        # use same shape and bounds as original phase space calculator
        # space_shape = (415, 252, 616)  # x, y, theta.
        space_shape = self.empty_space().shape  # x, y, theta.

        xbounds = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
        ybounds = (0, maze.arena_height)

        final_arr = np.empty(space_shape, bool)  # better to use space_shape[::-1] in terms of access speed
        thet_arr = np.linspace(0, 2 * np.pi, space_shape[2], False)

        # make the final array slice by slice
        for i, theta in enumerate(thet_arr):
            if not i % 50: print(f"{i}/{space_shape[2]}")
            final_arr[:, :, i] = self.calc_theta_slice(theta, space_shape[0], space_shape[1], xbounds, ybounds)


        self.space = self.shift_by_pi(final_arr) # somehow this was shifted by pi...


    def calc_theta_slice(self, theta, res_x, res_y, xbounds, ybounds):
        arr = np.ones((res_x, res_y), bool)
        im = Image.fromarray(arr)  # .astype('uint8')?
        draw = ImageDraw.Draw(im)

        s, c = np.sin(theta), np.cos(theta)
        rotation_mat = np.array(((c, -s), (s, c)))
        load_points = (rotation_mat@(self.load_points.T)).T

        for maze_edge in self.maze_edges:
            maze_edge = (self.maze_points[maze_edge[0]],
                         self.maze_points[maze_edge[1]])
            for load_edge in self.load_edges:
                load_edge = (load_points[load_edge[0]],
                             load_points[load_edge[1]])
                self.imprint_boundary(draw, arr.shape, load_edge, maze_edge, xbounds, ybounds)

        return np.array(im)  # type: ignore  # this is the canonical way to convert Image to ndarray

    @staticmethod
    def imprint_boundary(draw, shape, edge_1, edge_2, xbounds, ybounds):
        """
        Takes arr, and sets to 0 all pixels which intersect/lie inside the quad roughly describing
        the pixels which contain a point such that a shift by it causes the two edges to intersect

        @param draw: PIL ImageDraw object
        @param shape: the image shape (res_y, res_x)
        @param edge_1: first edge
        @param edge_2: second edge
        """

        # Reflected Binary Code~
        points = tuple(p + edge_2[0] for p in edge_1) + tuple(p + edge_2[1] for p in edge_1[::-1])

        # project into array space
        points = np.array(points)
        points[:, 0] -= xbounds[0];
        points[:, 0] *= shape[0] / (xbounds[1] - xbounds[0])
        points[:, 1] -= ybounds[0];
        points[:, 1] *= shape[1] / (ybounds[1] - ybounds[0])
        points += .5;
        points = points.astype(int)  # round to nearest integer

        draw.polygon(tuple(points[:, ::-1].flatten()), fill=0, outline=0)

    def new_fig(self):
        """
        Opening a new figure.
        :return:
        """
        fig = mlab.figure(figure=self.name, bgcolor=(1, 1, 1,), fgcolor=(0, 0, 0,), size=(800, 800))
        return fig

    def visualize_space(self, reduction: int = 1, fig=None, colormap: str = 'Greys', space: np.ndarray = None) -> None:
        """
        Visualize space using mayavi.
        :param reduction: if the PS should be displayed with reduced resolution in favor of run time
        :param fig: figure handle to plt the space in.
        :param colormap: colors used for plotting
        :param space: what space is supposed to be plotted. If none is given, self.space is used.
        """
        if fig is None and (self.fig is None or not self.fig.running):
            self.fig = self.new_fig()
        else:
            self.fig = fig

        x, y, theta = np.mgrid[self.extent['x'][0]:self.extent['x'][1]:self.pos_resolution * reduction,
                      self.extent['y'][0]:self.extent['y'][1]:self.pos_resolution * reduction,
                      self.extent['theta'][0]:self.extent['theta'][1]:self.theta_resolution * reduction,
                      ]

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (800, 160)
        if self.space is None:
            self.load_space()

        if space is None:
            space = np.array(self.space, dtype=int)
        else:
            space = np.array(space, dtype=int)

        if reduction > 1:
            space = self.reduced_resolution(space, reduction)

        def prune(array1, array2):
            """
            Prune the longer axis, for every axis, when dimensions of arrays are not equal
            :param array1: first array
            :param array2: second array
            :return: array1 and array2 with the same size.
            """
            for axis in range(array1.ndim):
                if array2.shape[axis] > array1.shape[axis]:
                    array2 = array2.take(indices=range(array1.shape[axis]), axis=axis)
                elif array2.shape[axis] < array1.shape[axis]:
                    array1 = array1.take(indices=range(array2.shape[axis]), axis=axis)
            return array1, array2

        if x.shape != space.shape:
            x, space = prune(x, space)
            y, space = prune(y, space)
            theta, space = prune(theta, space)

        cont = mlab.contour3d(x, y, theta,
                              space[:x.shape[0], :x.shape[1], :x.shape[2]],
                              opacity=0.08,  # 0.15
                              figure=self.fig,
                              colormap=colormap)

        cont.actor.actor.scale = [1, 1, self.average_radius]
        mlab.view(-90, 90)

    def iterate_space_index(self, mask=None) -> iter:
        """
        :param mask: If mask is given, only the unmasked areas are iterated over.
        :return: iterator over the indices_to_coords of self.space
        """
        if mask is None:
            for x_i in tqdm(range(self.space.shape[0])):
                for y_i, theta_i in itertools.product(range(self.space.shape[1]),
                                                      range(self.space.shape[2])):
                    yield x_i, y_i, theta_i
        else:
            for x_i in tqdm(tqdm(range(np.array(np.where(mask))[:, 0][0], np.array(np.where(mask))[:, -1][0]))):
                for y_i, theta_i in itertools.product(range(self.space.shape[1]),
                                                      range(self.space.shape[2])):
                    yield x_i, y_i, theta_i

    def iterate_coordinates(self, mask=None) -> iter:
        """
        :param mask: If mask is given, only the unmasked areas are iterated over.
        :return: iterator over the coords of self.space
        """
        if mask is None:
            x_iter = np.arange(self.extent['x'][0], self.extent['x'][1], self.pos_resolution)
            for x in tqdm(x_iter):
                for y in np.arange(self.extent['y'][0], self.extent['y'][1], self.pos_resolution):
                    for theta in np.arange(self.extent['theta'][0], self.extent['theta'][1], self.theta_resolution):
                        yield x, y, theta
        else:
            x0, y0, theta0 = np.array(np.where(mask))[:, 0]
            x1, y1, theta1 = np.array(np.where(mask))[:, -1]
            for x in tqdm(np.arange(self.indices_to_coords(x0, 0, 0)[0],
                                    self.indices_to_coords(x1, 0, 0)[0],
                                    self.pos_resolution)):
                for y in np.arange(self.indices_to_coords(0, y0, 0)[1],
                                   self.indices_to_coords(0, y1, 0)[1],
                                   self.pos_resolution):
                    for theta in np.arange(self.indices_to_coords(0, 0, theta0)[2],
                                           self.indices_to_coords(0, 0, theta1)[2],
                                           self.theta_resolution):
                        if mask[self.coords_to_indices(x, y, theta)]:
                            yield x, y, theta

    def iterate_neighbours(self, ix, iy, itheta) -> iter:
        """
        Iterate over all the neighboring pixels of a given pixel
        :param ix: index in 0 axis direction
        :param iy: index in 1 axis direction
        :param itheta: index in 2 axis direction
        :return: iterator over the neighbors
        """
        for dx, dy, dtheta in itertools.product([-1, 0, 1], repeat=3):
            _ix, _iy, _itheta = ix + dx, iy + dx, itheta + dtheta
            if ((_ix >= self.space.shape[0]) or (_ix < 0)
                    or (_iy >= self.space.shape[1]) or (_iy < 0)
                    or (_itheta >= self.space.shape[2]) or (_itheta < 0)
                    or ((dx, dy, dtheta) == (0, 0, 0))):
                continue
            yield _ix, _iy, _itheta

    # @staticmethod
    # def plot_trajectory(traj, color=(0, 0, 0)):
    #     mlab.plot3d(traj[0],
    #                 traj[1],
    #                 traj[2],
    #                 color=color, tube_radius=0.045, colormap='Spectral')
    #     mlab.points3d([traj[0, 0]], [traj[1, 0]], [traj[2, 0]])

    def save_space(self, directory: str = None) -> None:
        """
        Pickle the numpy array in given path, or in default path. If default directory exists, add a string for time, in
        order not to overwrite the old .pkl file.
        :param directory: Where you would like to save.
        """
        if not hasattr(self, 'space') and self.space is not None:
            self.calculate_space()
        if not hasattr(self, 'space_boundary') and self.space_boundary is not None:
            self.calculate_boundary()
        if directory is None:
            if os.path.exists(self.directory()):
                # now = datetime.now()
                # date_string = '_' + now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
                directory = self.directory(addition='')
            else:
                directory = self.directory()
        print('Saving ' + self.name + ' in path: ' + directory)
        pickle.dump((np.array(self.space, dtype=bool),
                     np.array(self.space_boundary, dtype=bool),
                     self.extent),
                    open(directory, 'wb'))

    def load_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        """
        directory = self.directory(point_particle=point_particle)
        if os.path.exists(directory):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(directory, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            self.calculate_boundary(point_particle=point_particle)
            self.save_space()
        return

    def _is_boundary_cell(self, x, y, theta) -> bool:
        if not self.space[x, y, theta]:
            return False
        for n_x, n_y, n_theta in self.iterate_neighbours(x, y, theta):
            if not self.space[n_x, n_y, n_theta]:
                return True
        return False

    def indices_to_coords(self, ix, iy, itheta) -> tuple:
        """
        Translating indices to coordinates
        :param ix: index in axis 0 direction
        :param iy: index in axis 1 direction
        :param itheta: index in axis 2 direction
        :return: set of coordinates (position, angle) that correspond to give indices of PS.
        """
        return (self.extent['x'][0] + ix * self.pos_resolution,
                self.extent['y'][0] + iy * self.pos_resolution,
                self.extent['theta'][0] + itheta * self.theta_resolution)

    def coords_to_index(self, axis: int, value):
        """
        Translating coords to index of axis
        :param axis: What axis is coordinate describing
        :param value:
        :return:
        """
        if value is None:
            return None
        resolut = {0: self.pos_resolution, 1: self.pos_resolution, 2: self.theta_resolution}[axis]
        value_i = min(int(np.round((value - list(self.extent.values())[axis][0]) / resolut)),
                      self.space.shape[axis] - 1)

        if value_i >= self.space.shape[axis] or value_i <= -1:
            print('check', list(self.extent.keys())[axis])
        return value_i

    def coords_to_indices(self, x: float, y: float, theta: float) -> tuple:
        """
        convert coordinates into indices_to_coords in PhaseSpace
        :param x: x position of CM in cm
        :param y: y position of CM in cm
        :param theta: orientation of axis in radian
        :return: (xi, yi, thetai)
        """
        return self.coords_to_index(0, x), self.coords_to_index(1, y), \
               self.coords_to_index(2, theta % (2 * np.pi))

    def empty_space(self) -> np.array:
        return np.zeros((int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
                         int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
                         int(np.ceil(
                             (self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))),
                        dtype=bool)

    def calculate_boundary(self, point_particle=False, mask=None) -> None:
        """
        Calculate the boundary of a given PhaseSpace.
        :param point_particle:
        :param mask: Where to calculate space (usefull just for testing)
        :return:
        """
        if self.space is None:
            self.calculate_space(mask=mask)
        self.space_boundary = self.empty_space()
        print("PhaseSpace: Calculating boundaries " + self.name)
        for ix, iy, itheta in self.iterate_space_index(mask=mask):
            if self._is_boundary_cell(ix, iy, itheta):
                self.space_boundary[ix, iy, itheta] = 1

    def draw(self, positions, angles, scale_factor: float = 0.5, color=(1, 0, 0)) -> None:
        """
        draw positions and angles in 3 dimensional phase space.
        """
        if np.array(positions).ndim == 1:
            positions = np.expand_dims(np.array(positions), axis=0)
            angles = np.expand_dims(np.array(angles), axis=0)
        # if self.VideoWriter is None:
        #     self.VideoWriter = cv2.VideoWriter('mayavi_Capture.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 20, (354, 400))
        angle = angles * self.average_radius
        mlab.points3d(positions[:, 0], positions[:, 1], angle,
                      figure=self.fig,
                      scale_factor=scale_factor,
                      color=color
                      )

    def draw_ind(self, indices: tuple, scale_factor: float = 0.2, color=None) -> None:
        """
        Draw single indices in a PS
        :param indices: indices of node to draw
        :param scale_factor: how much the PhaseSpace has been scaled
        """
        coords = self.indices_to_coords(*indices)
        self.draw(coords[:2], coords[2], scale_factor=scale_factor, color=color)

    def trim(self, borders: list) -> None:
        """
        Trim a phaseSpace down to size given by borders
        :param borders:
        :return:
        """
        [[x_min, x_max], [y_min, y_max]] = borders
        self.extent['x'] = (max(0, x_min), min(self.extent['x'][1], x_max))
        self.extent['y'] = (max(0, y_min), min(self.extent['y'][1], y_max))
        self.space = self.space[max(0, int(x_min / self.pos_resolution)):
                                min(int(x_max / self.pos_resolution) + 1, self.space.shape[0]),
                     max(0, int(y_min / self.pos_resolution)):
                     min(int(y_max / self.pos_resolution) + 1, self.space.shape[1]),
                     ]

    @staticmethod
    def dilate(space: np.array, radius: int) -> np.array:
        """
        dilate phase space
        :param radius: radius of dilation
        """
        print('Dilating space...')
        struct = np.ones([radius for _ in range(space.ndim)], dtype=bool)
        return np.array(ndimage.binary_dilation(space, structure=struct), dtype=bool)

    @staticmethod
    def erode(space, radius: int) -> np.array:
        """
        Erode phase space.
        We erode twice
        :param space: Actual space you want to erode
        :param radius: radius of erosion
        """
        print('Eroding space...')

        def erode_space(space, struct):
            return ndimage.binary_erosion(space, structure=struct)
            # return np.array(~ndimage.binary_erosion(~np.array(space, dtype=bool), structure=struct), dtype=bool)

        struct = np.ones([radius for _ in range(space.ndim)], dtype=bool)
        space1 = erode_space(space, struct)

        slice = int(space.shape[-1] / 2)
        space2 = erode_space(np.concatenate([space[:, :, slice:], space[:, :, :slice]], axis=2), struct)
        space2 = np.concatenate([space2[:, :, slice:], space2[:, :, :slice]], axis=2)

        return np.logical_or(space1, space2)

    def split_connected_components(self, space: np.array) -> tuple:
        """
        from self find connected components
        Take into account periodicity
        :param space: which space should be split
        :return: list of ps spaces, that have only single connected components
        """
        labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        stats = cc3d.statistics(labels)
        voxel_counts = [stats['voxel_counts'][label] for label in range(stats['voxel_counts'].shape[0])]

        max_cc_size = np.sort(voxel_counts)[-1] - 1  # this one is the largest, empty space
        min_cc_size = np.sort(voxel_counts)[-cc_to_keep - 2] - 1
        chosen_cc = np.where((max_cc_size > stats['voxel_counts']) & (stats['voxel_counts'] > min_cc_size))[0]
        assert chosen_cc.shape[0] == 10, 'We dont have the right number of connected components: ' + str(
            chosen_cc.shape[0])
        return chosen_cc, labels, stats['centroids']

    def extend_ps_states_to_eroded_space(self, ps_states):
        for ps_state in tqdm(ps_states):
            ps_state.distance = ps_state.calculate_distance(ps_state.space, self.space)

        distance_stack = np.stack([ps_state.distance for ps_state in ps_states], axis=3)
        for indices in self.iterate_space_index():
            if self.space[indices]:
                ps_state_to_add_to = np.argmin(distance_stack[indices])
                ps_states[ps_state_to_add_to].space[indices] = True

        for ps_state in tqdm(ps_states):
            ps_state.distance = None

        return ps_states

    def create_ps_states(self):
        # interesting_indices = (207, 175, 407), (207, 176, 408)  # human large
        if self.eroded_space is None:

            if self.solver == 'humanhand': # somehow otherwise the states will not properly be separated.
                directory = '\\\\phys-guru-cs\\ants\\Tabea\\PyCharm_Data\\AntsShapes\\Configuration_Spaces\\SPT\\' \
                            '_SPT_MazeDimensions_humanhand_pre_erosion.pkl'
                if os.path.exists(directory):
                    (space, _, _) = pickle.load(open(directory, 'rb'))

            else:
                space = self.space
            self.eroded_space = self.erode(space, radius=self.erosion_radius)
        chosen_cc, labels, centroids = self.split_connected_components(self.eroded_space)

        spaces = [np.bool_(labels == cc) for cc in chosen_cc]
        centroids = [centroids[cc] for cc in chosen_cc]

        ps_states = []
        given_names = []

        for space, centroid in zip(spaces, centroids):
            # self.visualize_space(space=space, reduction=4)
            connected = [saved_ps.try_to_connect_periodically(space, centroid) for saved_ps in ps_states]

            if not np.any(connected):
                name = self.name_for_state(space)
                if name in given_names:
                    raise ValueError()
                given_names.append(name)

                ps = PS_Area(self, space, name, centroid=centroid)
                ps_states.append(ps)

        self.ps_states = self.extend_ps_states_to_eroded_space(ps_states)
        self.correct_ps_states()

    def name_for_state(self, space):
        shape = self.space.shape
        if np.mean(np.where(space)[0])/shape[0] < 0.25:
            return 'a'
        if 0.3 < np.mean(np.where(space)[0])/shape[0] < 0.5 and (0 in np.where(space)[2] or shape[2]-1 in np.where(space)[2]):
            return 'b'
        if 0.3 < np.mean(np.where(space)[0])/shape[0] < 0.5 and shape[2]//2 in np.where(space)[2]:
            return 'c'
        if 0.5 < np.mean(np.where(space)[0])/shape[0] < 0.8 and (0 in np.where(space)[2] or shape[2]-1 in np.where(space)[2]):
            return 'f'
        if 0.5 < np.mean(np.where(space)[0])/shape[0] < 0.8 and shape[2]//2 in np.where(space)[2]:
            return 'g'
        if 0.75 < np.mean(np.where(space)[0])/shape[0]:
            return 'h'
        if np.mean(np.where(space)[2])/shape[2] < 0.5:
            return 'e'
        if 0.5 < np.mean(np.where(space)[2])/shape[2]:
            return 'd'
        raise ValueError


    def ps_name_dict(self):
        return {ps_state.name: i for i, ps_state in enumerate(self.ps_states)}

    def correct_ps_states(self):
        """
        States e and d are differently eroded for different sizes. Some leave the area elongated in y , some dont. This
        makes a difference in the state definition.
        """
        ps_name_dict = {ps_state.name: i for i, ps_state in enumerate(self.ps_states)}

        """
        Correction of e
        """

        g_indices = np.array(np.where(self.ps_states[ps_name_dict['g']].space))
        index = np.argmin(g_indices[2])
        g_ind = g_indices[:, index]

        b_indices_low = np.array(np.where(self.ps_states[ps_name_dict['b']].space[:, :, :self.space.shape[2]//2]))
        index = np.argmax(b_indices_low[2])
        b_ind = b_indices_low[:, index]

        transfer_from_c_to_e = np.array(np.where(self.ps_states[ps_name_dict['c']].space[:, :, :g_ind[2]]))
        for i in range(transfer_from_c_to_e.shape[1]):
            self.ps_states[ps_name_dict['c']].space[tuple(transfer_from_c_to_e[:, i])] = False
            self.ps_states[ps_name_dict['e']].space[tuple(transfer_from_c_to_e[:, i])] = True

        transfer_from_f_to_e = np.array(np.where(self.ps_states[ps_name_dict['f']].space[:, :, b_ind[2]:self.space.shape[2]//2]))
        transfer_from_f_to_e = transfer_from_f_to_e + np.array([[0, 0, b_ind[2]] for _ in range(transfer_from_f_to_e.shape[1])]).transpose()
        for i in range(transfer_from_f_to_e.shape[1]):
            self.ps_states[ps_name_dict['f']].space[tuple(transfer_from_f_to_e[:, i])] = False
            self.ps_states[ps_name_dict['e']].space[tuple(transfer_from_f_to_e[:, i])] = True

        """
        Correction of d
        """

        b_indices_high = np.array(np.where(self.ps_states[ps_name_dict['b']].space[:, :, self.space.shape[2]//2:]))
        b_indices_high = b_indices_high + np.array([[0, 0, self.space.shape[2]//2] for _ in range(b_indices_high.shape[1])]).transpose()
        index = np.argmin(b_indices_high[2])
        b_ind = b_indices_high[:, index]

        g_indices = np.array(np.where(self.ps_states[ps_name_dict['g']].space))
        index = np.argmax(g_indices[2])
        g_ind = g_indices[:, index]

        transfer_from_c_to_d = np.array(np.where(self.ps_states[ps_name_dict['c']].space[:, :, g_ind[2]:]))
        transfer_from_c_to_d = transfer_from_c_to_d + np.array([[0, 0, g_ind[2]] for _ in range(transfer_from_c_to_d.shape[1])]).transpose()
        for i in range(transfer_from_c_to_d.shape[1]):
            self.ps_states[ps_name_dict['c']].space[tuple(transfer_from_c_to_d[:, i])] = False
            self.ps_states[ps_name_dict['d']].space[tuple(transfer_from_c_to_d[:, i])] = True

        transfer_from_f_to_d = np.array(np.where(self.ps_states[ps_name_dict['f']].space[:, :, self.space.shape[2]//2:b_ind[2]]))
        transfer_from_f_to_d = transfer_from_f_to_d + np.array([[0, 0, self.space.shape[2]//2] for _ in range(transfer_from_f_to_d.shape[1])]).transpose()
        for i in range(transfer_from_f_to_d.shape[1]):
            self.ps_states[ps_name_dict['f']].space[tuple(transfer_from_f_to_d[:, i])] = False
            self.ps_states[ps_name_dict['d']].space[tuple(transfer_from_f_to_d[:, i])] = True


        self.ps_states[ps_name_dict['e']].clean()
        self.ps_states[ps_name_dict['d']].clean()

        self.visualize_space(space=self.ps_states[ps_name_dict['e']].space)
        self.visualize_space(space=self.ps_states[ps_name_dict['d']].space)

class PS_Area(ConfigSpace_Maze):
    def __init__(self, ps: ConfigSpace_Maze, space: np.array, name: str, centroid=None):
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape, geometry=ps.geometry)
        self.space: np.array = space
        self.fig = ps.fig
        self.name: str = name
        self.distance: np.array = None
        self.centroid = centroid

    def get_indices(self):
        return list(map(tuple, np.array(np.where(self.space)).transpose()))

    def clean(self, num_cc=1):
        labels, number_cc = cc3d.connected_components(self.space, connectivity=6, return_N=True)
        stats = cc3d.statistics(labels)
        while len(stats['voxel_counts']) > 1 + num_cc:
            self.space[labels == np.argmin(stats['voxel_counts'])] = False
            labels, number_cc = cc3d.connected_components(self.space, connectivity=6,
                                                          return_N=True)
            stats = cc3d.statistics(labels)

    def try_to_connect_periodically(self, space, centroid) -> bool:
        # if this is part of a another ps that is split by 0 or 2pi
        border_bottom = np.any(self.space[:, :, 0])
        border_top = np.any(self.space[:, :, -1])

        if (border_bottom and not border_top) or (border_top and not border_bottom):
            if len(np.where(np.abs(centroid[0] - self.centroid[0]) < 1)[0]) > 0:
                self.extend_by_periodic(space)
                return True
        return False

    def extend_by_periodic(self, space):
        self.space = np.logical_or(self.space, space)
        self.centroid[-1] = 0


class PS_Mask(ConfigSpace):
    def __init__(self, space):
        space = np.zeros_like(space, dtype=bool)
        super().__init__(space, name='mask')

    @staticmethod
    def paste(wall: np.array, block: np.array, loc: tuple) -> np.array:
        """
        :param wall:
        :param block:
        :param loc: the location of the wall, that should be in the (0, 0) position of the block
        """

        def paste_slices(tup) -> tuple:
            pos, w, max_w = tup
            wall_min = max(pos, 0)
            wall_max = min(pos + w, max_w)
            block_min = -min(pos, 0)
            block_max = max_w - max(pos + w, max_w)
            block_max = block_max if block_max != 0 else None
            return slice(wall_min, wall_max), slice(block_min, block_max)

        loc_zip = zip(loc, block.shape, wall.shape)
        wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
        wall[wall_slices] = block[block_slices]
        return wall

    def add_circ_mask(self, radius: int, indices: tuple):
        """
        because of periodic boundary conditions in theta, we first roll the array, then assign, then unroll
        add a circular mask to
        """

        def circ_mask() -> np.array:
            x, y, theta = np.ogrid[(-1) * radius: radius + 1, (-1) * radius: radius + 1, (-1) * radius: radius + 1]
            mask = np.array(x ** 2 + y ** 2 + theta ** 2 <= radius ** 2)
            return mask

        loc = (-radius + indices[0], -radius + indices[1], 0)
        self.paste(self.space, circ_mask(), loc)
        self.space = np.roll(self.space, -radius + indices[2], axis=2)


class Node:
    def __init__(self, indices):
        self.indices: tuple = indices

    def draw(self, ps):
        """
        Draw a node in PhaseSpace
        :param ps: PhaseSpace
        :return:
        """
        ps.draw_ind(self.indices)

    def find_closest_state(self, ps_states: list) -> int:
        """
        :return: name of the ps_state closest to indices_to_coords, chosen from ps_states
        """
        for radius in range(1, 50):
            print(radius)
            ps_mask = PS_Mask(ps_states[0])
            ps_mask.add_circ_mask(radius, self.indices)

            for ps_state in ps_states:
                if ps_state.overlapping(ps_mask):
                    print(ps_state.name)
                    # self.visualize_space()
                    # ps_state.visualize_space(self.fig)
                    # ps_mask.visualize_space(self.fig, colormap='Oranges')
                    return ps_state.name
        return 0

    def find_closest_states(self, ps_states: list, N: int = 1) -> list:
        """
        :param N: how many of the closest states do you want to find?
        :param ps_states: how many of the closest states do you want to find?
        :return: name of the closest PhaseSpace
        """

        state_order = []  # carries the names of the closest states, from closest to farthest

        for i in range(N):
            closest = self.find_closest_state(ps_states)
            state_order.append(closest)
            [ps_states.remove(ps_state) for ps_state in ps_states if ps_state.name == closest]
        return state_order


class ConfigSpace_Labeled(ConfigSpace_Maze):
    """
    This class stores configuration space for a piano_movers problem in a 3 dim array.
    Axis 0 = x direction
    Axis 1 = y direction
    Axis 0 = x direction
    Every element of self.space carries on of the following indices_to_coords of:
    - '0' (not allowed)
    - A, B...  (in self.eroded_space), where N is the number of states
    - n_1 + n_2 where n_1 and n_2 are in (A, B...).
        n_1 describes the state you came from.
        n_2 describes the state you are
    """

    def __init__(self, solver, size, shape, geometry, ps: ConfigSpace_Maze = None):
        if ps is None:
            ps = ConfigSpace_Maze(solver, size, shape, geometry, name='')
            ps.load_space()
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape, geometry=geometry)
        self.space = ps.space  # True, if there is collision. False, if it is an allowed configuration

        self.eroded_space = None
        self.erosion_radius = self.erosion_radius_default()
        self.ps_states = self.centroids = None
        self.space_labeled = None

    def load_eroded_labeled_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle. Load both eroded space, and ps_states, centroids and everything
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        """
        directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius)

        if os.path.exists(directory):
            print('Loading labeled from ', directory, '...')
            self.eroded_space, self.ps_states, self.space_labeled = pickle.load(open(directory, 'rb'))
            if len(self.ps_states) != cc_to_keep:
                print('Wrong number of cc')
        else:
            if self.ps_states is None:
                self.create_ps_states()
                # pickle.dump(self.ps_states, open('ps_states.pkl', 'wb'))

            # self.visualize_states(reduction=5)
            self.label_space()
            self.save_labeled()
        print('Finished loading')

    def load_labeled_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculafted.
        """
        directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius, small=True)

        if os.path.exists(directory):
            print('Loading labeled from ', directory, '.')
            # self.space_labeled = pickle.load(open(path, 'rb'))
            self.space_labeled = pickle.load(open(directory, 'rb'))
        else:
            self.load_eroded_labeled_space()

    def check_labels(self) -> list:
        """
        I want to check, whether there are labels, that I don't want to have.
        :return: list of all the labels that are not labels I wanted to have
        """
        return [label for label in np.unique(self.space_labeled) if label not in
                states + forbidden_transition_attempts + allowed_transition_attempts]

    # def label_space_slow(self) -> None:
    #     """
    #     label each node in PhaseSpace.space with a list
    #     """
    #     self.space_labeled = np.zeros([*self.space.shape, 2])
    #
    #     def calculate_label_slow(indices_to_coords: tuple) -> list:
    #         """
    #         Finds the label for the coordinates x, y and theta
    #         :return: integer or list
    #         """
    #         # everything not in self.space.
    #         if self.space[indices_to_coords]:
    #             return [0, np.NaN]
    #
    #         # everything in self.ps_states.
    #         for i, ps in enumerate(self.ps_states):
    #             # [not self.ps_states[i].space[indices_to_coords] for i in range(len(self.ps_states))]
    #             if ps.space[indices_to_coords]:
    #                 return [i, np.NaN]
    #
    #         # in eroded space
    #         return Node(indices_to_coords).find_closest_states(self.ps_states, N=2)
    #
    #     def label_slice(x_i):
    #         matrix = np.zeros((self.space.shape[1], self.space.shape[2], 2))
    #         for y_i, theta_i in itertools.product(range(self.space.shape[1]), range(self.space.shape[2])):
    #             matrix[y_i, theta_i] = calculate_label_slow((x_i, y_i, theta_i))
    #         print('finished slice ', str(x_i))
    #         return matrix
    #
    #     start_time = datetime.now()
    #     label_slice(100)
    #     print('Duration: {}'.format(datetime.now() - start_time))
    #
    #     matrices = Parallel(n_jobs=4)(delayed(label_slice)(x_i) for x_i in range(self.space.shape[0]))
    #     self.space_labeled = np.stack(matrices, axis=0)

    def visualize_states(self, fig=None, reduction: int = 1) -> None:
        """

        :param fig: mylab figure reference
        :param colormap: What color do you want the available states to appear in?
        :param reduction: What amount of reduction?
        :return:
        """
        if self.ps_states is None:
            self.load_eroded_labeled_space()
        if self.fig is None or not self.fig.running:
            self.visualize_space(reduction=reduction)
        else:
            self.fig = fig

        print('Draw states')
        if len(self.ps_states) != cc_to_keep:
            print('Wrong number of cc')

        colors = itertools.cycle(['Reds', 'Purples', 'Greens'])

        for ps_state, colormap in tqdm(zip(self.ps_states, colors)):
            ps_state.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction)
            mlab.text3d(*(np.array(ps.indices_to_coords(*ps_state.centroid)) * [1, 1, self.average_radius]),
                        ps_state.name,
                        scale=self.scale_of_letters(reduction))

    def scale_of_letters(self, reduction):
        return {'Large': 1, 'Medium': 0.5, 'Small Far': 0.2, 'Small Near': 0.2, 'Small': 0.2,
                'L': 0.5, 'XL': 1, 'M': 0.5, 'S': 0.25, '': 1}[self.size] * reduction

    def visualize_transitions(self, fig=None, reduction: int = 1) -> None:
        """

        :param fig: mylab figure reference
        :param reduction: What amount of reduction?
        :return:
        """
        if self.fig is None or not self.fig.running:
            self.visualize_space(reduction=reduction)

        else:
            self.fig = fig

        if self.space_labeled is None:
            self.load_labeled_space()

        print('Draw transitions')
        transitions = [trans for trans in np.unique(self.space_labeled) if len(trans) > 1]
        for label, colormap in tqdm(zip(transitions, itertools.cycle(['Reds', 'Purples', 'Greens']))):
            space = np.array(self.space_labeled == label, dtype=bool)
            centroid = self.indices_to_coords(*np.array(np.where(space))[:, 0])
            self.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction, space=space)
            mlab.text3d(*(a * b for a, b in zip(centroid, [1, 1, self.average_radius])), label,
                        scale=self.scale_of_letters(reduction))

    def save_labeled(self, directory=None) -> None:
        """
        Save the labeled PhaseSpace in directory
        :param directory: where to save PhaseSpace
        :param date_string:
        :return:
        """
        if directory is None:
            directory = self.directory(point_particle=False, erosion_radius=self.erosion_radius, addition='')

        if os.path.exists(directory):
            now = datetime.now()
            date_string = now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
            directory = self.directory(point_particle=False, erosion_radius=self.erosion_radius, addition=date_string)

        print('Saving ' + self.name + ' in path: ' + directory)
        pickle.dump((self.eroded_space, self.ps_states, self.space_labeled), open(directory, 'wb'))

        # Actually, I don't really need all this information.  self.space_labeled should be enough
        directory = self.directory(point_particle=False, erosion_radius=self.erosion_radius, addition='', small=True)

        if os.path.exists(directory):
            now = datetime.now()
            date_string = now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
            directory = self.directory(point_particle=False, erosion_radius=self.erosion_radius, addition=date_string,
                                       small=True)

        print('Saving reduced in ' + self.name + ' in path: ' + directory)
        pickle.dump(self.space_labeled, open(directory, 'wb'))
        print('Finished saving')

    def erosion_radius_default(self) -> int:
        """
        for ant XL, the right radius of erosion is 0.9cm. To find this length independent of scaling for every system,
        we use this function.
        :return:
        """
        default = self.coords_to_indices(0, (self.extent['y'][1] / 21.3222222), 0)[1]
        if self.size in ['Small Far', 'Small Near'] and self.solver == 'human':
            return default + 4
        if self.solver == 'ant':
            if self.size == 'S':
                return default + 4
            return default + 3
        if self.solver == 'humanhand':
            return default - 2
        return default
        # return int(np.ceil(self.coords_to_indices(0, 0.9, 0)[0]))

    def max_distance_for_transition(self) -> int:
        """
        :return: maximum distance (in pixels in CS) so that it is noted as transition area
        """
        maze = Maze(solver=self.solver, size=self.size, shape=self.shape, geometry=self.geometry)
        if self.shape == 'SPT':
            distance_cm = (maze.slits[1] - maze.slits[0]) / 2
        else:
            distance_cm = maze.exit_size / 2
        return self.coords_to_index(0, distance_cm)
        # return (self.coords_to_index(0, distance_cm) + self.erosion_radius * 2)

    def add_false_connections(self) -> np.array:
        ps_name_dict = self.ps_name_dict()
        space_with_false_connections = copy(self.space)

        state1, state2 = 'bf'
        axis_connect = 0
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices()
                     if inds[1]==int(self.space.shape[1]/2)], key=lambda x: x[2])[-1]
        s2 = (np.min(np.where(self.ps_states[ps_name_dict[state2]].space[:, s1[1], s1[2]])), s1[1], s1[2])
        space_with_false_connections[s1[0] - 1:s2[0] + 1, s1[1], s1[2]] = True

        state1, state2 = 'cg'
        axis_connect = 0
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices()
                               if inds[2]==int(self.space.shape[2]/2) and inds[1]==int(self.space.shape[1]/2)],
                              key=lambda x: x[2])[-1]
        s2 = (np.min(np.where(self.ps_states[ps_name_dict[state2]].space[:, s1[1], s1[2]])),
                        s1[1], s1[2])
        space_with_false_connections[s1[0] - 1:s2[0] + 1, s1[1], s1[2]] = True

        def connect_dots():
            space_with_false_connections[s1[0] - 1: s2[0] + 1, s1[1], s1[2]] = True
            space_with_false_connections[s2[0], min(s1[1], s2[1]) - 1: max(s1[1], s2[1]) + 1, s1[2]] = True
            space_with_false_connections[s2[0], s2[1], min(s1[2], s2[2]) - 1: max(s1[2], s2[2]) + 1] = True

        state1, state2 = 'bd'
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices() if inds[2] > 100],
                              key = lambda x: x[1])[0]
        s2 = sorted([inds for inds in self.ps_states[ps_name_dict[state2]].get_indices()],
                              key=lambda x: x[2])[-1]
        connect_dots()

        state1, state2 = 'be'
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices() if inds[2] < 200],
                              key = lambda x: x[1])[-1]
        s2 = sorted([inds for inds in self.ps_states[ps_name_dict[state2]].get_indices()],
                              key=lambda x: x[2])[0]
        connect_dots()

        state1, state2 = 'eg'
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices()],
                              key=lambda x: x[2])[-1]
        s2 = sorted([inds for inds in self.ps_states[ps_name_dict[state2]].get_indices() if inds[2] < self.space.shape[2]//2],
                              key=lambda x: x[1])[-1]
        connect_dots()

        state1, state2 = 'dg'
        s1 = sorted([inds for inds in self.ps_states[ps_name_dict[state1]].get_indices()],
                              key=lambda x: x[2])[0]
        s2 = sorted([inds for inds in self.ps_states[ps_name_dict[state2]].get_indices() if inds[2] > self.space.shape[2]//2],
                              key=lambda x: x[1])[0]
        connect_dots()
        return space_with_false_connections

    def label_space(self) -> None:
        """
        Calculate the labeled space.
        :return:
        """
        # interesting_indices = (207, 175, 407) # human large
        if self.ps_states is None:
            self.create_ps_states()
        ps_name_dict = {i: ps_state.name for i, ps_state in enumerate(self.ps_states)}

        print('Calculating distances from every node for ', str(len(self.ps_states)), ' different states in', self.name)
        space_with_false_connections = self.add_false_connections()
        # dilated_space = self.dilate(self.space, self.erosion_radius_default())

        for ps_state in tqdm(self.ps_states):
            ps_state.distance = ps_state.calculate_distance(ps_state.space, space_with_false_connections)

        # distance_stack_original = np.stack([ps_state.distance for ps_state in self.ps_states], axis=3)
        distance_stack = copy(np.stack([ps_state.distance for ps_state in self.ps_states], axis=3))
        far_away = distance_stack > self.max_distance_for_transition()
        distance_stack[far_away] = np.inf

        self.space_labeled = np.zeros_like(self.space, dtype=np.dtype('U2'))
        print('Iterating over every node and assigning label')
        # self.assign_label(interesting_indices, distance_stack, ps_name_dict)
        [self.assign_label(indices, distance_stack, ps_name_dict) for indices in self.iterate_space_index()]
        self.fix_edges_labeling()

    def assign_label(self, ind: tuple, distance_stack: np.array, ps_name_dict: dict):
        """

        """
        # everything not in self.space.
        if not self.space[ind]:
            self.space_labeled[ind] = '0'
            return

        # # everything in self.ps_states.
        # for i, ps_state in enumerate(self.ps_states):
        #     if ps_state.space[ind]:
        label = ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack[ind])[:2]
                         if distance_stack[ind].data[ii] < np.inf])
        if label in forbidden_transition_attempts + allowed_transition_attempts:
            self.space_labeled[ind] = label
        else:
            self.space_labeled[ind] = [ps_name_dict[ii] for ii in np.argsort(distance_stack[ind])[:1]][0]

        if len(self.space_labeled[ind]) == 0:
            raise ValueError(ind)
        return
        # in eroded space
        # self.visualize_states()
        # self.draw_ind(indices)
        # self.space_labeled[ind] = ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack[ind])[:2]
        #                                    if distance_stack[ind].data[ii] < np.inf])

        # if len(self.space_labeled[ind]) == 0:
        #     self.space_labeled[ind] = ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack_original[ind])[:2]])

    def find_closest_state(self, index: list, border=10, last_label=None) -> str:
        """
        :return: name of the ps_state closest to indices_to_coords, chosen from ps_states
        """
        index_theta = index[2]
        found_close_state = False
        border = 5
        while not found_close_state:
            if border > 100:
                raise ValueError('Cant find closest state: ' + str(index))
            if index_theta - border < 0 or index_theta + border > self.space_labeled.shape[2]:
                cut_out = np.concatenate([self.space_labeled[max(0, index[0] - border):index[0] + border,
                                          max(0, index[1] - border):index[1] + border,
                                          (index_theta - border) % self.space_labeled.shape[2]:],
                                          self.space_labeled[max(0, index[0] - border):index[0] + border,
                                          max(0, index[1] - border):index[1] + border,
                                          0:(index_theta + border) % self.space_labeled.shape[2]]
                                          ], axis=-1)
            else:
                cut_out = self.space_labeled[max(0, index[0] - border):index[0] + border,
                          max(0, index[1] - border):index[1] + border,
                          index[2] - border:index[2] + border]

            distances = {}
            states = np.unique(cut_out).tolist()

            if '0' in states:
                states.remove('0')

            if len(states) > 0:
                found_close_state = True
            else:
                border += 10
        if len(states) == 1:
            return states[0]
        for state in states:
            distances[state] = self.calculate_distance(cut_out == state, np.ones(shape=cut_out.shape, dtype=bool))[border, border, border]
        return min(distances, key=distances.get)

    def fix_edges_labeling(self):
        """
        for some reason some single states for larger x are called a. I make them empty states, here.
        """
        problematic_states = {'a': {'human': {'Large': 200, 'Medium': 140, 'Small Far': 250},
                                    'humanhand': {'': 250},
                                    'ant': {'XL': 180, 'L': 180, 'M': 150, 'S': 250}},
                              # 'ca': {'human': {'Large': None, 'Medium': None, 'Small Far': None},
                              #        'ant': {'XL': None, 'L': None, 'M': None, 'S': None}}
                              }
        problematic_states['ab'] = {'humanhand': {'': 270},
                                    'human': {'Large': 220, 'Medium': 160, 'Small Far': 270},
                                    'ant': {'XL': 200, 'L': 200, 'M': 170, 'S': 270}}

        for problematic_state, border in problematic_states.items():
            border_index = border[self.solver][self.size]
            if border_index is not None:
                indices = np.where(self.space_labeled == problematic_state)
                false = np.where(indices[0] > border_index)[0]
                delete = np.stack(indices)[:, false]

                print(self.space_labeled.shape)
                print('Deleting', delete.shape[1], problematic_state, 'elements')
                for i in range(delete.shape[1]):
                    to_change = tuple(delete[:, i])
                    if to_change[0] > self.space_labeled.shape[0]:
                        print(to_change)
                    # print(cs_labeled.space_labeled[to_change])
                    self.space_labeled[tuple(to_change)] = '0'



if __name__ == '__main__':
    shape = 'SPT'
    geometries_to_change = {('humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')): ['']}

    geometries = {
        ('ant', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')): ['XL', 'L', 'M', 'S'],
        ('human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')): ['Large', 'Medium', 'Small Far'],
        ('humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')): ['']
        }

    for (solver, geometry), sizes in list(geometries_to_change.items()):
        for size in sizes:
            print(solver, size)
            # ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, geometry=geometry)
            # ps.load_space()
            # ps.space = ps.shift_by_pi(ps.space)
            # ps.visualize_space(reduction=4)
            # ps.save_space()
            # DEBUG = 1

            ps = ConfigSpace_Labeled(solver=solver, size=size, shape=shape, geometry=geometry)
            ps.load_eroded_labeled_space()
            ps.visualize_transitions(reduction=2)
            DEBUG = 1


            # ps.eroded_space = ps.shift_by_pi(ps.eroded_space)
            # ps.visualize_space(space=ps.eroded_space, reduction=4)
            #
            # for i in range(len(ps.ps_states)):
            #     ps.ps_states[i].space = ps.shift_by_pi(ps.ps_states[i].space)
            #
            # ps.space_labeled = None
            # ps.label_space()
            #
            #
            # # ps.visualize_states(reduction=4)
            # # ps.visualize_transitions(reduction=4)
            #
            #
            # ps.save_labeled()
            # # ps.save_labeled()
            # DEBUG = 1

            # TODO: I think there is still a problem with 'ca' and 'ac' in the human medium CS.

