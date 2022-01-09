import math
from mayavi import mlab
import numpy as np
from copy import copy


class Node_ind:
    """
    DeLiteNode Node
    """

    def __init__(self, xi, yi, thetai, conf_space_shape, average_radius):
        self.xi = xi
        self.yi = yi
        self.thetai = thetai  # this takes values between 0 and 2*np.pi
        self.path_xi = []
        self.path_yi = []
        self.path_thetai = []  # this takes values between 0 and 2*np.pi
        self.parent = None
        self.conf_space_shape = conf_space_shape
        self.average_radius = average_radius

    def __str__(self):
        return 'Node with ind = {}, {}, {}'.format(*self.ind())

    def ind(self):
        if self.thetai is None:
            return self.xi, self.yi, self.thetai
        else:
            return self.xi, self.yi, self.thetai % self.conf_space_shape[2]

    def connected(self, conf_space):
        nn = []
        for xi, yi, thetai in self.iterate_surroundings(conf_space):
            if not conf_space.space[xi, yi, thetai]:
                nn.append((xi, yi, thetai))

        if self.ind() not in nn:
            print('you came from false node')
        else:
            nn.remove(self.ind())

        #  What if you get into a Sackgasse, because you are dont know Phase space fully,
        #  and then you cannot step on your old path...  I think, I have to take out the condition,
        #  that you cannot step on your old path...
        # parent = self.parent
        # while parent is not None:
        #     if parent.ind() in nn:
        #         nn.remove(parent.ind())
        #     parent = parent.parent

        return nn

    def iterate_surroundings(self, conf_space):
        # I want the walker to prefer not to wobble back and forth, so I sort the order I want to ...
        thetai = [(self.thetai - 1) % conf_space.space.shape[2],
                  self.thetai,
                  (self.thetai + 1) % conf_space.space.shape[2]]
        xi = [max(0, self.xi - 1), self.xi, min(conf_space.space.shape[0] - 1, self.xi + 1)]
        yi = [max(0, self.yi - 1), self.yi, min(conf_space.space.shape[1] - 1, self.yi + 1)]

        yield xi[1], yi[1], thetai[1]
        yield xi[0], yi[1], thetai[1]
        yield xi[2], yi[1], thetai[1]
        yield xi[1], yi[0], thetai[1]
        yield xi[1], yi[2], thetai[1]
        yield xi[1], yi[1], thetai[0]
        yield xi[1], yi[1], thetai[2]

        yield xi[0], yi[0], thetai[0]
        yield xi[0], yi[1], thetai[0]
        yield xi[0], yi[2], thetai[0]
        yield xi[0], yi[0], thetai[1]
        yield xi[0], yi[2], thetai[1]
        yield xi[0], yi[0], thetai[2]
        yield xi[0], yi[1], thetai[2]
        yield xi[0], yi[2], thetai[2]

        yield xi[1], yi[0], thetai[0]
        yield xi[1], yi[2], thetai[0]
        yield xi[1], yi[0], thetai[2]
        yield xi[1], yi[2], thetai[2]

        yield xi[2], yi[0], thetai[0]
        yield xi[2], yi[1], thetai[0]
        yield xi[2], yi[2], thetai[0]
        yield xi[2], yi[0], thetai[1]
        yield xi[2], yi[2], thetai[1]
        yield xi[2], yi[0], thetai[2]
        yield xi[2], yi[1], thetai[2]
        yield xi[2], yi[2], thetai[2]

    def iterate_surroundings2(self, conf_space):
        for thetai in [self.thetai,
                       (self.thetai - 1) % conf_space.space.shape[2],
                       (self.thetai + 1) % conf_space.space.shape[2]]:
            for xi in [self.xi, max(0, self.xi - 1), min(conf_space.space.shape[0] - 1, self.xi + 1)]:
                for yi in [self.yi, max(0, self.yi - 1), min(conf_space.space.shape[1] - 1, self.yi + 1)]:
                    yield xi, yi, thetai

    @staticmethod
    def surrounding(conf_space, ind, radius=1):
        rolled = np.roll(conf_space.space, -(ind[2] - radius), axis=2)
        return np.array(rolled[ind[0] - radius:ind[0] + radius + 1,
                               ind[1] - radius:ind[1] + radius + 1,
                               : 2 * radius + 1],
                        dtype=bool)

    def coord(self, conf_space):
        return conf_space.indices_to_coords(*self.ind())

    def distance(self, node, conf_space):
        coo_self = self.coord(conf_space)
        coo_node = node.coord(conf_space)
        return np.sqrt((coo_node[0] - coo_self[0]) ** 2 +
                       (coo_node[1] - coo_self[1]) ** 2 +
                       (((coo_node[2] - coo_self[2] + np.pi) % (2 * np.pi) - np.pi) * self.average_radius) ** 2)

    def calc_distance_and_angles(self, to_node, conf_space):
        coo_self = self.coord(conf_space)
        coo_to_node = to_node.coord(conf_space)

        dx = coo_to_node[0] - coo_self[0]
        dy = coo_to_node[1] - coo_self[1]
        dtheta = (((coo_to_node[2] - coo_to_node[2] + np.pi) % (2 * np.pi) - np.pi) * self.average_radius)

        r = np.linalg.norm([dx, dy, dtheta])
        azimut = math.atan2(dy, dx)  # 0 to 2pi
        polar = math.acos(dtheta / r)  # 0 to pi

        return r, azimut, polar

    def get_nearest_node(self, node_list):
        dlist = [self.distance(node) for node in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    def draw_node(self, conf_space, fig=None, scale_factor=0.2, color=(0, 0, 0)):
        coo = self.coord(conf_space)
        mlab.points3d(coo[0], coo[1], coo[2] * self.average_radius,
                      figure=fig,
                      scale_factor=scale_factor,
                      color=color,
                      )

    def draw_line(self, node, conf_space, fig=None, line_width=0.2, color=(0, 0, 0)):
        coo_self = self.coord(conf_space)
        coo_node = node.coord(conf_space)

        if abs(coo_node[2] - coo_self[2]) > np.pi:
            if coo_node[2] > coo_self[2]:
                upper_node, lower_node = copy(coo_node), copy(coo_self)
            else:
                upper_node, lower_node = copy(coo_self), copy(coo_node)

            d = lower_node[2] % (2 * np.pi) - upper_node[2]
            A = lower_node[2] % (2 * np.pi) - np.pi
            half_way_x = (coo_node[0] - coo_self[0]) * A / d + coo_self[0]
            half_way_y = (coo_node[1] - coo_self[1]) * A / d + coo_self[1]

            mlab.plot3d([upper_node[0], half_way_x],
                        [upper_node[1], half_way_y],
                        [upper_node[2] * self.average_radius, 2 * np.pi * self.average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )
            mlab.plot3d([half_way_x, lower_node[0]],
                        [half_way_y, lower_node[1]],
                        [0, lower_node[2] * self.average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )

        else:
            mlab.plot3d([coo_self[0], coo_node[0]], [coo_self[1], coo_node[1]],
                        [coo_self[2] * self.average_radius, coo_node[2] * self.average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )
