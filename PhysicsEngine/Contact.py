from scipy.spatial import cKDTree
from Setup.MazeFunctions import BoxIt
import numpy as np
from Setup.Load import loops
from Analysis.GeneralFunctions import flatten

# maximum distance between fixtures to have a contact (in cm)
distance_upper_bound = 0.04


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def maze_corners(maze):
    corners = [[0, 0],
               [0, maze.arena_height],
               [maze.slits[-1] + 20, maze.arena_height],
               [maze.slits[-1] + 20, 0],
               ]
    return corners + list(np.resize(maze.slitpoints, (16, 2)))


def possible_configuration(load, maze) -> bool:
    """
    this function takes a list of corners (lists have to list rectangles in sets of 4 corners)
    and checks, whether a rectangle from the load overlaps with a rectangle from the maze boundary
    :return: bool, whether the shape intersects the maze
    """
    # first check, whether we are in the middle of the maze, where there is no need for heavy calculations
    approximate_extent = np.max([2 * np.max(np.abs(np.array(list(fix.shape)))) for fix in load.fixtures])

    if approximate_extent + 0.1 < load.position.x < min(maze.slits) - approximate_extent - 0.1 and \
            approximate_extent + 0.1 < load.position.y < maze.arena_height - approximate_extent - 0.1:
        return True

    elif max(maze.slits) + approximate_extent + 0.1 < load.position.x:
        return True

    # if we are close enough to a boundary then we have to calculate all the vertices.
    load_corners = flatten(loops(load))
    maze_corners1 = maze_corners(maze)
    for load_NumFixture in range(int(len(load_corners) / 4)):
        load_vertices_list = load_corners[load_NumFixture * 4:(load_NumFixture + 1) * 4] \
                             + [load_corners[load_NumFixture * 4]]

        for maze_NumFixture in range(int(len(maze_corners1) / 4)):
            maze_vertices_list = maze_corners1[maze_NumFixture * 4:(maze_NumFixture + 1) * 4] \
                                 + [maze_corners1[maze_NumFixture * 4]]
            for i in range(4):
                for ii in range(4):
                    if intersect(load_vertices_list[i], load_vertices_list[i + 1],
                                 maze_vertices_list[ii], maze_vertices_list[ii + 1]):
                        return False
    return True
    # return np.any([f.TestPoint(load.position) for f in maze.body.fixtures])


def contact_loop_experiment(load, maze) -> list:
    """
    :return: list of all the points in world coordinates where the load is closer to the maze than distance_upper_bound.
    """
    edge_points = contact = []
    load_vertices = loops(load)

    for load_vertice in load_vertices:
        edge_points = edge_points + BoxIt(load_vertice, distance_upper_bound).tolist()

    load_tree = cKDTree(edge_points)
    in_contact = load_tree.query(maze.slitTree.data, distance_upper_bound=distance_upper_bound)[1] < \
                 load_tree.data.shape[0]

    if np.any(in_contact):
        contact = contact + maze.slitTree.data[np.where(in_contact)].tolist()
    return contact
