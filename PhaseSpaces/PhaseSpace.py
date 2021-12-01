import numpy as np
import itertools
import pickle
from mayavi import mlab
from PhysicsEngine.Contact import contact_loop_phase_space
from copy import copy
import os
from Setup.Maze import Maze
from Directories import PhaseSpaceDirectory, ps_path
from Analysis.PathLength import resolution
from scipy import ndimage
import cc3d

traj_color = (1.0, 0.0, 0.0)
start_end_color = (0.0, 0.0, 0.0)
scale = 5


# TODO: fix the display in this module
# TODO: fix the x.winner attribute

# I want the resolution (in cm) for x and y and archlength to be all the same.


class PhaseSpace(object):
    def __init__(self, solver:str, size:str, shape:str, name="", new2021:bool=False):
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

    def calculate_space(self, new2021: bool = False, point_particle=False, screen=None):
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
                space[self.coords_to_indexes(x, y, theta)] = np.any(contact_loop_phase_space(load, maze))
            return space

        # iterate using parallel processing
        # if parallel:
        #     n_jobs = 5
        #     x_index_first = int(np.floor(self.space.shape[0]/n_jobs))
        #     split_list = [(i * x_index_first, (i + 1) * x_index_first) for i in range(n_jobs-1)]
        #     split_list.append((split_list[-1][-1], self.space.shape[0]))
        #
        #     # we should be able to paralyze this process...
        #     # results = ps_calc(*split_list[0])
        #     import multiprocessing as mp
        #     pool = mp.Pool(mp.cpu_count())
        #     # k = ps_calc(self, load, maze, 0, 2)
        #     results = [pool.apply(ps_calc, args=(copy(self), copy(load), copy(maze), copy(x0), copy(x1)))
        #                for x0, x1 in [(0, 2), (2, 4)]]
        #     # results = Parallel(n_jobs=5)(delayed(ps_calc)(x0, x1) for x0, x1 in [(0, 2), (2, 4)])
        #     # results = Parallel(n_jobs=5)(delayed(ps_calc)(x0, x1) for x0, x1 in split_list)
        #     self.space = np.concatenate(results, axis=0)

        self.space = ps_calc(0, self.space.shape[0])
        return

    def new_fig(self):
        fig = mlab.figure(figure=self.name, bgcolor=(1, 1, 1,), fgcolor=(0, 0, 0,), size=(800, 800))
        return fig

    def visualize_space(self, fig=None, colormap='Greys') -> None:
        if fig is not None:
            self.fig = fig
        else:
            self.fig = self.new_fig()
        x, y, theta = np.mgrid[self.extent['x'][0]:self.extent['x'][1]:self.pos_resolution,
                               self.extent['y'][0]:self.extent['y'][1]:self.pos_resolution,
                               self.extent['theta'][0]:self.extent['theta'][1]:self.theta_resolution,
                               ]

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (800, 160)
        cont = mlab.contour3d(x, y, theta,
                              self.space[:x.shape[0], :x.shape[1], :x.shape[2]],
                              opacity=0.15,
                              figure=self.fig,
                              colormap=colormap)

        cont.actor.actor.scale = [1, 1, self.average_radius]
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

    def iterate_coordinates(self, x0: int = 0, x1: int = -1):
        r"""
        param x0: indice to start with
        param x1: indice to end with
        :return: iterator
        """
        x_iter = np.arange(self.extent['x'][0], self.extent['x'][1], self.pos_resolution)[x0:x1]
        for x in x_iter:
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

    @staticmethod
    def plot_trajectory(traj, color=(0, 0, 0)):
        mlab.plot3d(traj[0],
                    traj[1],
                    traj[2],
                    color=color, tube_radius=0.045, colormap='Spectral')
        mlab.points3d([traj[0, 0]], [traj[1, 0]], [traj[2, 0]])

    def save_space(self, path='SLT.pkl'):
        print('Saving ' + self.name + ' in path: ' + path)
        pickle.dump((self.space, self.space_boundary, self.extent), open(path, 'wb'))

    def load_space(self, point_particle: bool = False, new2021: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param path: path, where the pickle to be loaded is stored.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        :param new2021: for the small Special T, used in 2021, the maze had different geometry than before.
        """
        path = ps_path(self.size, self.shape, point_particle=point_particle, new2021=new2021)
        if os.path.exists(path):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(path, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            self.calculate_boundary(point_particle=point_particle, new2021=new2021)
            self.save_space(path=path)
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
        erode phase space
        :param radius: radius of erosion
        """
        struct = np.ones([radius for _ in range(self.space.ndim)], dtype=bool)
        self.space = np.array(~ndimage.binary_erosion(~np.array(self.space, dtype=bool), structure=struct), dtype=int)

    def split_connected_components(self, min=10) -> (list, list):
        """
        from self find connected components, and return a list of ps spaces, that have only single connected components.
        """
        pss = centroids = []
        labels, number_cc = cc3d.connected_components(np.invert(np.array(self.space, dtype=bool)),
                                                      connectivity=6, return_N=True)
        stats = cc3d.statistics(labels)

        for label, centroid, voxel_count in zip(range(1, number_cc), stats['centroids'], stats['voxel_counts']):
            if voxel_count > min:
                ps = PhaseSpace(self.solver, self.size, self.shape)
                ps.space = np.int8(labels == label)
                pss.append(ps)
                centroids.append(self.indexes_to_coords(*np.floor(centroid)))
        return pss, centroids


if __name__ == '__main__':
    shape = 'H'
    size = 'XL'
    point_particle = False
    solver = 'ant'

    name = size + '_' + shape

    if point_particle:
        name = name + '_pp'

    path = os.path.join(PhaseSpaceDirectory, solver, name + ".pkl")
    ps = PhaseSpace(solver, size, shape, name=name)
    ps.load_space()
    ps.visualize_space()
    k = 1
