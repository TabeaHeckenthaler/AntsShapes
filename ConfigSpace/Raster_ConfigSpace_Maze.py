import numpy as np
from PIL import Image, ImageDraw

from Analysis.GeneralFunctions import flatten
from Setup.Load import loops


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
    points[:, 0] -= xbounds[0]; points[:, 0] *= shape[0]/(xbounds[1] - xbounds[0])
    points[:, 1] -= ybounds[0]; points[:, 1] *= shape[1]/(ybounds[1] - ybounds[0])
    points += .5; points = points.astype(int)  # round to nearest integer

    draw.polygon(tuple(points[:, ::-1].flatten()), fill=0, outline=0)


class Raster_ConfigSpace_Maze:
    def __init__(self, maze):
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

    # Note: this is unoptimal - we duplicate points and use unnecessary edges. Could be like, 15% faster with
    # smarter edge placement and such

    def __call__(self, theta, res_x, res_y, xbounds, ybounds):
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
                imprint_boundary(draw, arr.shape, load_edge, maze_edge, xbounds, ybounds)

        return np.array(im)  # type: ignore  # this is the canonical way to convert Image to ndarray