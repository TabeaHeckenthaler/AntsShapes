import numpy as np
import itertools
import pickle
from mayavi import mlab
from PhysicsEngine.Contact import Contact_loop2
from Setup.Load import Load
from copy import copy
import os
from Setup.Load import average_radius, getLoadDim
from Setup.Maze import Maze
from progressbar import progressbar
from trajectory import home
from Analysis_Functions.Pathlength import resolution

traj_color = (1.0, 0.0, 0.0)
start_end_color = (0.0, 0.0, 0.0)
scale = 5

# I want the resolution (in cm) for x and y and archlength to be all the same.

ps_dir = home + '\\PhaseSpaces'


class PhaseSpace(object):

    def __init__(self, solver, size, shape, name="",):
        """
        :param board_coords:
        :param load_coords:
        # :param pos_resolution: load replacement resolution (in in the coords units)
        # :param theta_resolution: theta replace
        # :param x_range: tuple of the x-space range, in the coords units
        # :param y_range: tuple of the y-space range, in the coords units
        """
        maze = Maze(size=size, shape=shape, solver=solver)

        self.name = name
        self.solver = solver
        self.shape = shape
        self.size = size

        x_range = (0, maze.slits[-1] + max(getLoadDim(maze.solver, maze.shape, maze.size)) + 1)
        y_range = (0, maze.arena_height)

        self.extent = {'x': x_range,
                       'y': y_range,
                       'theta': (0, np.pi * 2)}

        self.pos_resolution = self.extent['y'][1] / self.number_of_points()['y']
        self.theta_resolution = 2 * np.pi / self.number_of_points()['theta']

        self.space = None  # True, if there is a collision with the wall
        self.space_boundary = None
        # self._initialize_maze_edges()

    def number_of_points(self):
        # x_num = np.ceil(self.extent['x'][1]/resolution)
        y_num = np.ceil(self.extent['y'][1] / resolution(self.size, self.solver))
        theta_num = np.ceil(self.extent['theta'][1] * average_radius(self.size, self.shape, self.solver) / resolution(self.size, self.solver))
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

    def calculate_space(self, point_particle=False):
        maze = Maze(size=self.size, shape=self.shape, solver=self.solver)
        load = Load(maze, point_particle=point_particle)

        # initialize 3d map for the phase_space
        self.space = np.ones((int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
                              int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
                              int(np.ceil(
                                  (self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))))
        print("PhaseSpace: Calculating space " + self.name)

        # lines_stat, circles = screen(maze, [], [])

        for x, y, theta in self.iterate_coordinates():
            load.position, load.angle = [x, y], float(theta)
            self.space[self.coords_to_indexes(x, y, theta)] = np.any(Contact_loop2(load, maze))
            # if not self.space[self.coords_to_indexes(x, y, theta)]:
            #     Display_renew(0, maze)
            #     _, _, _ = Pygame_EventManager(0, maze, load, [], copy(lines_stat), circles, pause=False)

    def visualize_space(self, name):
        vis_space = copy(self.space)

        x, y, theta = np.mgrid[self.extent['x'][0]:self.extent['x'][1]:self.pos_resolution,
                      self.extent['y'][0]:self.extent['y'][1]:self.pos_resolution,
                      self.extent['theta'][0]:self.extent['theta'][1]:self.theta_resolution,
                      ]

        fig = mlab.figure(figure=name,
                          bgcolor=(1, 1, 1,),
                          fgcolor=(0, 0, 0,),
                          size=(400, 400))

        cont = mlab.contour3d(x, y, theta,
                              vis_space[:x.shape[0], :x.shape[1], :x.shape[2]],
                              opacity=0.15,
                              figure=fig,
                              colormap='gray')

        cont.actor.actor.scale = [1, 1, average_radius(self.size, self.shape, self.solver)]

        """ to get theta """
        ax = mlab.axes(xlabel="x",
                       ylabel="y",
                       zlabel="theta",
                       line_width=2,
                       ranges=[self.extent['x'][0], self.extent['x'][1],
                               self.extent['y'][0], self.extent['y'][1],
                               self.extent['theta'][0], self.extent['theta'][1],
                               ],
                       )

        ax.axes.label_format = '%.2f'
        ax.label_text_property.font_family = 'times'
        return fig

    def iterate_coordinates(self):
        for x in progressbar(np.arange(self.extent['x'][0], self.extent['x'][1], self.pos_resolution)):
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
    def plot_trajectory(self, traj, color=(0, 0, 0)):
        mlab.plot3d(traj[0],
                    traj[1],
                    traj[2],
                    color=color, tube_radius=0.045, colormap='Spectral')
        mlab.points3d([traj[0, 0]], [traj[1, 0]], [traj[2, 0]])

    def save_space(self, path='SLT.pkl'):
        pickle.dump((self.space, self.space_boundary, self.extent), open(path, 'wb'))

    def load_space(self, path=os.path.join('C:\\Users\\tabea\\PycharmProjects\\AntsShapes'
                                           '\\PhaseSpaces\\ant\\XL_SPT.pkl'),
                   point_particle=False):
        if os.path.exists(path):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(path, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            self.calculate_boundary(point_particle=point_particle)
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

    def calculate_boundary(self, point_particle=False):
        if self.space is None:
            self.calculate_space(point_particle=point_particle)
        self.space_boundary = np.zeros(
            (int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))))
        for ix, iy, itheta in self.iterate_space_index():
            if self._is_boundary_cell(ix, iy, itheta):
                self.space_boundary[ix, iy, itheta] = 1

    def draw_trajectory(self, fig, positions, angles, scale_factor=0.02, **kwargs):
        angle = angles * average_radius(self.size, self.shape, self.solver)
        mlab.points3d(positions[:, 0], positions[:, 1], angle,
                      figure=fig,
                      scale_factor=scale_factor,
                      **kwargs
                      )
        return fig

    def trim(self, borders):
        [[x_min, x_max], [y_min, y_max]] = borders
        self.extent['x'] = (max(0, x_min), min(self.extent['x'][1], x_max))
        self.extent['y'] = (max(0, y_min), min(self.extent['y'][1], y_max))
        self.space = self.space[max(0, int(x_min / self.pos_resolution)):
                                min(int(x_max / self.pos_resolution) + 1, self.space.shape[0]),
                     max(0, int(y_min / self.pos_resolution)):
                     min(int(y_max / self.pos_resolution) + 1, self.space.shape[1]),
                     ]
        return

# ps = PhaseSpace('ant', 'XL', 'H', name='XL_H')
# ps.load_space()
# ps.visualize_space(ps.name)
