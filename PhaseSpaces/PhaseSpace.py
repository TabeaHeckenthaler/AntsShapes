import numpy as np
import itertools
import pickle
from mayavi import mlab
from PhysicsEngine.Contact import contact_loop_phase_space
import os
import itertools
from Setup.Maze import Maze
from Directories import PhaseSpaceDirectory, ps_path
from Analysis.PathLength import resolution
from scipy import ndimage
import cc3d
from datetime import datetime
import string
from joblib import Parallel, delayed
from skfmm import distance
from tqdm import tqdm
from itertools import groupby
from copy import copy

traj_color = (1.0, 0.0, 0.0)
start_end_color = (0.0, 0.0, 0.0)
scale = 5


# TODO: fix the display in this module
# TODO: fix the x.winner attribute

# I want the resolution (in cm) for x and y and archlength to be all the same.


class PhaseSpace(object):
    def __init__(self, solver: str, size: str, shape: str, name="", new2021: bool = False):
        """
        :param board_coords:
        :param load_coords:
        # :param pos_resolution: load replacement resolution (in in the coords units)
        # :param theta_resolution: theta replace
        # :param x_range: tuple of the x-space range, in the coords units
        # :param y_range: tuple of the y-space range, in the coords units
        # :param new2021: whether these are new dimensions for SSPT maze after we made it smaller in 2021
        """
        maze = Maze(size=size, shape=shape, solver=solver, new2021=new2021)

        if len(name) == 0:
            name = size + '_' + shape

        self.name = name
        self.solver = solver
        self.shape = shape
        self.size = size

        x_range = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
        y_range = (0, maze.arena_height)

        self.extent = {'x': x_range,
                       'y': y_range,
                       'theta': (0, np.pi * 2)}
        self.average_radius = maze.average_radius()

        self.pos_resolution = self.extent['y'][1] / self.number_of_points()['y']
        self.theta_resolution = 2 * np.pi / self.number_of_points()['theta']

        self.space = None  # True, if there is a collision with the wall
        self.space_boundary = None
        self.fig = None

        # self.monitor = {'left': 3800, 'top': 160, 'width': 800, 'height': 800}
        # self.VideoWriter = cv2.VideoWriter('mayavi_Capture.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 20,
        #                                    (self.monitor['width'], self.monitor['height']))
        # self._initialize_maze_edges()

    def __add__(self, PhaseSpace2):
        FinalPS = PhaseSpace(self.solver, self.size, self.shape, name=self.name)
        FinalPS.space = np.logical_and(self.space, PhaseSpace2)
        return FinalPS

    def iter_inds(self) -> iter:
        for x_i, y_i, theta_i in itertools.product(range(self.space.shape[0]),
                                                   range(self.space.shape[1]),
                                                   range(self.space.shape[2])):
            yield x_i, y_i, theta_i

    def number_of_points(self):
        # x_num = np.ceil(self.extent['x'][1]/resolution)
        y_num = np.ceil(self.extent['y'][1] / resolution(self.size, self.solver))
        theta_num = np.ceil(
            self.extent['theta'][1] * self.average_radius / resolution(self.size, self.solver))
        return {'x': None, 'y': y_num, 'theta': theta_num}

    def initialize_maze_edges(self):
        """
        set x&y edges to 0 in order to define the maze boundaries (helps the visualization)
        :return:
        """
        self.space[0, :, :] = 1
        self.space[-1, :, :] = 1
        self.space[:, 0, :] = 1
        self.space[:, -1, :] = 1

    def calculate_space(self, new2021: bool = False, point_particle=False, screen=None, parallel=False):
        # TODO: implement point particles
        maze = Maze(size=self.size, shape=self.shape, solver=self.solver, new2021=new2021)
        load = maze.bodies[-1]

        # initialize 3d map for the phase_space
        self.space = np.ones((int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
                              int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
                              int(np.ceil(
                                  (self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))))
        print("PhaseSpace: Calculating space " + self.name)

        # how to iterate over phase space
        def ps_calc(x0, x1):
            """
            param x0: index of x array to start with
            param x1: index of x array to end with
            :return: iterator"""
            space = np.zeros([x1 - x0, self.space.shape[1], self.space.shape[2]])
            for x, y, theta in self.iterate_coordinates(x0=x0, x1=x1):
                load.position, load.angle = [x, y], float(theta)
                coord = self.coords_to_indexes(x, y, theta)
                space[coord] = contact_loop_phase_space(load, maze)
            return space

        # how to iterate over phase space
        def ps_calc_parallel(iterate: iter, space: np.array, load, maze, coords):
            """
            param iterate: index of x array to start with
            param space: index of x array to end with
            :return: space
            """
            for (x, y, theta), coord in zip(iterate, coords):
                load.position, load.angle = [x, y], float(theta)
                space[coord] = contact_loop_phase_space(load, maze)
            return space

        # iterate using parallel processing
        if parallel:
            n_jobs = 5
            x_index_first = int(np.floor(self.space.shape[0] / n_jobs))
            split_list = [(i * x_index_first, (i + 1) * x_index_first) for i in range(n_jobs - 1)]
            split_list.append((split_list[-1][-1], self.space.shape[0]))

            list_of_jobs = []
            for x0, x1 in split_list:  # split_list
                space = np.zeros([x1 - x0, self.space.shape[1], self.space.shape[2]])
                coords = [self.coords_to_indexes(x, y, theta)
                          for x, y, theta in self.iterate_coordinates(x0=0, x1=x1 - x0)]
                iterator = self.iterate_coordinates(x0=x0, x1=x1)
                list_of_jobs.append(delayed(ps_calc_parallel)(iterator, space, copy(load), copy(maze), coords))
                # m = ps_calc_parallel(iterator, space, copy(load), copy(maze))

            matrices = Parallel(n_jobs=n_jobs, prefer='threads')(list_of_jobs)
            self.space = np.concatenate(matrices, axis=0)

        else:
            self.space = ps_calc(0, self.space.shape[0])
        return

    def new_fig(self):
        fig = mlab.figure(figure=self.name, bgcolor=(1, 1, 1,), fgcolor=(0, 0, 0,), size=(800, 800))
        return fig

    def visualize_space(self, fig=None, colormap='Greys') -> None:
        if self.fig is None or not self.fig.running:
            self.fig = self.new_fig()
        else:
            self.fig = fig

        x, y, theta = np.mgrid[self.extent['x'][0]:self.extent['x'][1]:self.pos_resolution,
                      self.extent['y'][0]:self.extent['y'][1]:self.pos_resolution,
                      self.extent['theta'][0]:self.extent['theta'][1]:self.theta_resolution,
                      ]

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (800, 160)
        if self.space is None:
            self.load_space()
        space = np.array(self.space, dtype=int)
        cont = mlab.contour3d(x, y, theta,
                              space[:x.shape[0], :x.shape[1], :x.shape[2]],
                              opacity=0.15,
                              figure=self.fig,
                              colormap=colormap)

        cont.actor.actor.scale = [1, 1, self.average_radius]
        mlab.view(-90, 90)
        # ax = mlab.axes(xlabel="x",
        #                ylabel="y",
        #                zlabel="theta",
        #                line_width=2,
        #                ranges=[self.extent['x'][0], self.extent['x'][1],
        #                        self.extent['y'][0], self.extent['y'][1],
        #                        self.extent['theta'][0], self.extent['theta'][1],
        #                        ],
        #                )
        #
        # ax.axes.label_format = '%.2f'
        # ax.label_text_property.font_family = 'times'

    def iterate_coordinates(self, x0: int = 0, x1: int = -1) -> iter:
        r"""
        param x0: index to start with
        param x1: index to end with
        :return: iterator
        """
        x_iter = np.arange(self.extent['x'][0], self.extent['x'][1], self.pos_resolution)[x0:x1]
        for x in tqdm(x_iter):
            for y in np.arange(self.extent['y'][0], self.extent['y'][1], self.pos_resolution):
                for theta in np.arange(self.extent['theta'][0], self.extent['theta'][1], self.theta_resolution):
                    yield x, y, theta

    def iterate_neighbours(self, ix, iy, itheta):
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

    def save_space(self, path=None):
        if path is None:
            now = datetime.now()
            date_string = now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
            path = self.size + '_' + self.shape + '_' + date_string + '.pkl'
        print('Saving ' + self.name + ' in path: ' + path)
        pickle.dump((self.space, self.space_boundary, self.extent), open(path, 'wb'))

    def load_space(self, point_particle: bool = False, new2021: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        :param new2021: for the small Special T, used in 2021, the maze had different geometry than before.
        """
        path = ps_path(self.size, self.shape, solver=self.solver, point_particle=point_particle, new2021=new2021)
        if os.path.exists(path):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(path, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            self.calculate_boundary(point_particle=point_particle, new2021=new2021)
            self.save_space()
        return

    def _is_boundary_cell(self, x, y, theta):
        if not self.space[x, y, theta]:
            return False
        for n_x, n_y, n_theta in self.iterate_neighbours(x, y, theta):
            if not self.space[n_x, n_y, n_theta]:
                return True
        return False

    def indexes_to_coords(self, ix, iy, itheta):
        return (self.extent['x'][0] + ix * self.pos_resolution,
                self.extent['y'][0] + iy * self.pos_resolution,
                self.extent['theta'][0] + itheta * self.theta_resolution)

    def coords_to_indexes(self, x, y, theta):
        if x is not None:
            x = min(int(np.round((x - self.extent['x'][0]) / self.pos_resolution)), self.space.shape[0] - 1)
            if x >= self.space.shape[0] or x <= -1:
                print('check x')
        if y is not None:
            y = min(int(np.round((y - self.extent['y'][0]) / self.pos_resolution)), self.space.shape[1] - 1)
            if y >= self.space.shape[1] or y <= -1:
                print('check y')
        if theta is not None:
            theta = min(int(np.round((theta % (2 * np.pi) -
                                      self.extent['theta'][0]) / self.theta_resolution)),
                        self.space.shape[2])
            if theta >= self.space.shape[2] or theta <= -1:
                print('check theta')
        return x, y, theta

    def iterate_space_index(self):
        for ix in range(self.space.shape[0]):
            for iy in range(self.space.shape[1]):
                for itheta in range(self.space.shape[2]):
                    yield ix, iy, itheta

    def calculate_boundary(self, new2021=False, point_particle=False):
        if self.space is None:
            self.calculate_space(point_particle=point_particle, new2021=new2021)
        self.space_boundary = np.zeros(
            (int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))))
        for ix, iy, itheta in self.iterate_space_index():
            if self._is_boundary_cell(ix, iy, itheta):
                self.space_boundary[ix, iy, itheta] = 1

    def draw(self, positions, angles, scale_factor: float = 0.5, color=(1, 0, 0)):
        """
        draw positions and angles in 3 dimensional phase space
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

    def trim(self, borders):
        [[x_min, x_max], [y_min, y_max]] = borders
        self.extent['x'] = (max(0, x_min), min(self.extent['x'][1], x_max))
        self.extent['y'] = (max(0, y_min), min(self.extent['y'][1], y_max))
        self.space = self.space[max(0, int(x_min / self.pos_resolution)):
                                min(int(x_max / self.pos_resolution) + 1, self.space.shape[0]),
                     max(0, int(y_min / self.pos_resolution)):
                     min(int(y_max / self.pos_resolution) + 1, self.space.shape[1]),
                     ]

    def dilate(self, radius: int = 8) -> None:
        """
        dilate phase space
        :param radius: radius of dilation
        """
        struct = np.ones([radius for _ in range(self.space.ndim)], dtype=bool)
        self.space = np.array(~ndimage.binary_dilation(~np.array(self.space, dtype=bool), structure=struct), dtype=int)

    def erode(self, radius: int = 8) -> None:
        """
        Erode phase space.
        We erode twice
        :param radius: radius of erosion
        """

        def erode_space(space, struct):
            return np.array(~ndimage.binary_erosion(~np.array(space, dtype=bool), structure=struct), dtype=int)

        struct = np.ones([radius for _ in range(self.space.ndim)], dtype=bool)
        space1 = erode_space(self.space, struct)

        slice = int(self.space.shape[-1] / 2)
        space2 = erode_space(np.concatenate([self.space[:, :, slice:], self.space[:, :, :slice]], axis=2), struct)
        space2 = np.concatenate([space2[:, :, slice:], space2[:, :, :slice]], axis=2)

        self.space = np.array(np.logical_and(space1, space2), dtype=int)

    def split_connected_components(self, min=10) -> (list, list):
        """
        from self find connected components
        Take into account periodicity
        :return: list of ps spaces, that have only single connected components
        """
        ps_states = []
        letters = list(string.ascii_lowercase)
        centroids = np.empty((0, 3))
        labels, number_cc = cc3d.connected_components(np.invert(np.array(self.space, dtype=bool)),
                                                      connectivity=6, return_N=True)
        stats = cc3d.statistics(labels)

        for label in range(1, number_cc):
            if stats['voxel_counts'][label] > min:
                ps = PS_Area(self, np.int8(labels == label), letters.pop(0))

                # if this is part of a another ps that is split by 0 or 2pi
                centroid = np.array(self.indexes_to_coords(*np.floor(stats['centroids'][label])))

                border_bottom = np.any(ps.space[:, :, 0])
                border_top = np.any(ps.space[:, :, -1])
                if (border_bottom and not border_top) or (border_top and not border_bottom):
                    index = np.where(centroid[0] == centroids[:, 0])[0]
                    if len(index) > 0:
                        ps_states[index[0]].space = np.array(np.logical_or(ps_states[index[0]].space, ps.space),
                                                             dtype=int)
                        centroids[index[0]][-1] = 0
                    else:
                        ps_states.append(ps)
                        centroids = np.vstack([centroids, centroid])
                else:
                    ps_states.append(ps)
                    centroids = np.vstack([centroids, centroid])
        return ps_states, centroids


class PS_Area(PhaseSpace):
    def __init__(self, ps: PhaseSpace, space: np.array, name: str):
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape)
        self.space: np.array = space
        self.fig = ps.fig
        self.name: str = name
        self.distance: np.array = None

    def overlapping(self, ps_area):
        return np.any(self.space[ps_area.space])

    def calculate_distance(self):
        self.distance = distance(np.array((~np.array(self.space, dtype=bool)), dtype=int), periodic=(0, 0, 1))


class PS_Mask(PS_Area):
    def __init__(self, ps):
        space = np.zeros(ps.space.shape, dtype=bool)
        super().__init__(ps, space, 'mask')

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
        self.space = np.roll(self.space, indices[2], axis=2)


class Node:
    def __init__(self, indices):
        self.indices: tuple = indices

    def draw(self, ps):
        ps.draw(ps.indexes_to_coords(*self.indices)[:2], ps.indexes_to_coords(*self.indices)[2])

    def find_closest_states(self, ps_states: list, N: int = 1) -> list:
        """
        :param N: how many of the closest states do you want to find?
        :param ps_states: how many of the closest states do you want to find?
        :return: name of the closest PhaseSpace
        """

        def find_closest_state() -> int:  # TODO: how to speed up the process?
            """
            :return: name of the ps_state closest to indices, chosen from ps_states
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

        state_order = []  # carries the names of the closest states, from closest to farthest

        for i in range(N):
            closest = find_closest_state()
            state_order.append(closest)
            [ps_states.remove(ps_state) for ps_state in ps_states if ps_state.name == closest]
        return state_order


class PhaseSpace_Labeled(PhaseSpace):
    """
    This class stores configuration space for a piano_movers problem in a 3 dim array.
    Axis 0 = x direction
    Axis 1 = y direction
    Axis 0 = x direction
    Every element of self.space carries on of the following indices of:
    - '0' (not allowed)
    - A, B...  (in self.eroded_space), where N is the number of states
    - n_1 + n_2 where n_1 and n_2 are in (A, B...).
        n_1 describes the state you came from.
        n_2 describes the state you are
    """

    def __init__(self, ps: PhaseSpace, ps_eroded: np.array, ps_states: list, erosion_radius: int):
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape)
        self.space = ps.space  # 1, if there is collision. 0, if it is an allowed configuration
        self.ps_states = ps_states
        self.eroded_space = ps_eroded.space
        self.erosion_radius = erosion_radius
        self.space_labeled = None

    def load_space(self, point_particle: bool = False, new2021: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        :param new2021: for the small Special T, used in 2021, the maze had different geometry than before.
        """
        path = ps_path(self.size, self.shape, point_particle=point_particle, new2021=new2021,
                       erosion_radius=self.erosion_radius)

        if os.path.exists(path):
            self.space_labeled = pickle.load(open(path, 'rb'))
        else:
            self.label_space()
            self.save_labeled()

    # def label_space_slow(self) -> None:
    #     """
    #     label each node in PhaseSpace.space with a list
    #     """
    #     self.space_labeled = np.zeros([*self.space.shape, 2])
    #
    #     def calculate_label_slow(indices: tuple) -> list:
    #         """
    #         Finds the label for the coordinates x, y and theta
    #         :return: integer or list
    #         """
    #         # everything not in self.space.
    #         if self.space[indices]:
    #             return [0, np.NaN]
    #
    #         # everything in self.ps_states.
    #         for i, ps in enumerate(self.ps_states):
    #             # [not self.ps_states[i].space[indices] for i in range(len(self.ps_states))]
    #             if ps.space[indices]:
    #                 return [i, np.NaN]
    #
    #         # in eroded space
    #         return Node(indices).find_closest_states(self.ps_states, N=2)
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

    def save_labeled(self, path=None) -> None:
        if path is None:
            path = ps_path(self.size, self.shape, self.solver, point_particle=False, new2021=False,
                           erosion_radius=self.erosion_radius)  # TODO have different kwargs
        print('Saving ' + self.name + ' in path: ' + path)
        pickle.dump(self.space_labeled, open(path, 'wb'))

    def label_space(self) -> None:
        [ps_state.calculate_distance() for ps_state in tqdm(self.ps_states)]
        distance_stack = np.stack([ps_state.distance for ps_state in self.ps_states], axis=3)
        ps_name_dict = {i: ps_state.name for i, ps_state in enumerate(self.ps_states)}

        def calculate_label() -> str:
            # everything not in self.space.
            if self.space[indices]:
                return '0'

            # everything in self.ps_states.
            for i, ps in enumerate(self.ps_states):
                if ps.space[indices]:
                    return ps.name

            # in eroded space
            return ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack[indices])[:2]])

        self.space_labeled = np.zeros([*self.space.shape], dtype=np.dtype('U2'))

        for indices in self.iter_inds():
            self.space_labeled[indices] = calculate_label()
        return

    def label_trajectory(self, x):
        labels = [self.space_labeled[self.coords_to_indexes(*coords)] for coords in x.iterate_coords()]
        return labels

    @staticmethod  # maybe better to be part of a new class
    def reduces_labels(labels):
        return [''.join(ii[0]) for ii in groupby([tuple(label) for label in labels])]


if __name__ == '__main__':
    shape = 'H'
    size = 'XL'
    point_particle = False
    solver = 'ant'

    name = size + '_' + shape

    if point_particle:
        name = name + '_pp'

    ps = PhaseSpace(solver, size, shape, name=name)
    ps.load_space()
    ps.visualize_space()
    k = 1
