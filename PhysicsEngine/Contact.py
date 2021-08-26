from scipy.spatial import cKDTree
from Setup.MazeFunctions import BoxIt
import numpy as np
from Setup.Maze import Maze, maze_corners
from Analysis_Functions.Velocity import velocity_x
from Setup.Load import Loops
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop
from Setup.Load import Load
from Analysis_Functions.usefull_stuff import flatten

# maximum distance between fixtures to have a contact (in cm)
distance_upper_bound = 0.04


def find_contact(x, display=False):
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
    my_load = Load(my_maze, position=x.position[0])
    contact = []
    running, pause = True, False

    # to find contact in entire experiment
    if display:
        screen = Display_screen(my_maze=my_maze, caption=x.filename)

    i = 0
    while i < len(x.frames):
        x.step(my_load, i)  # update the position of the load (very simple function, take a look)

        if not pause:
            contact.append(Contact_loop(my_load, my_maze))
            i += 1

        if display:
            """Option 1"""
            # more simplistic, you are just renewing the screen, and displaying the objects
            Display_renew(screen)
            Display_loop(my_load, my_maze, screen, points=contact[-1])

            """Option 2"""
            # if you want to be able to pause the display, use this command:
            # running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=pause, points=contact[-1])

    if display:
        Display_end()
    return contact


def theta(r):
    [x, y] = r
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0 and y != 0:
        return np.sign(y) * np.pi / 2


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def Contact_loop2(load, maze):
    # this function takes a list of corners (lists have to list rectangles in sets of 4 corners)
    # and checks, whether a rectangle from the load overlaps with a rectangle from the

    # first check, whether we are in the middle of the maze, where there is no need for heavy calculations
    # approximate_extent = np.max([2 * np.max(np.abs(np.array(list(fix.shape)))) for fix in load.fixtures])
    #
    # if approximate_extent + 0.1 < load.position.x < min(maze.slits) - approximate_extent - 0.1 and \
    #         approximate_extent + 0.1 < load.position.y < maze.arena_height - approximate_extent - 0.1:
    #     return False
    #
    # elif max(maze.slits) + approximate_extent + 0.1 < load.position.x:
    #     return False

    # if we are close enough to a boundary then we have to calculate all the vertices.
    load_corners = flatten(Loops(load))
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
                        return True

    return np.any([f.TestPoint(load.position) for f in maze.body.fixtures])


def Contact_loop(my_load, my_maze):
    contact = []
    edge_points = []
    load_vertices = Loops(my_load)

    for load_vertice in load_vertices:
        edge_points = edge_points + BoxIt(load_vertice, distance_upper_bound).tolist()

    load_tree = cKDTree(edge_points)
    in_contact = load_tree.query(my_maze.slitTree.data, distance_upper_bound=distance_upper_bound)[1] < \
                 load_tree.data.shape[0]

    if np.any(in_contact):
        contact = contact + my_maze.slitTree.data[np.where(in_contact)].tolist()
    return contact


def find_Impact_Points(x, my_maze, *args, **kwargs):
    x, contact = x.play(1, 'contact', *args, *kwargs)
    wall_contacts = np.where([len(con) > 0
                              # x component close to the exit wall
                              and con[0][0] > my_maze.slits[0] - 1
                              #  and (abs(con[0][1] - my_maze.arena_height / 2 - my_maze.exit_size / 2) < 2
                              #       or abs(con[0][1] - my_maze.arena_height / 2 + my_maze.exit_size / 2) < 2)
                              for con in contact])[0]

    # only if its not a to short contact!
    wall_contacts = [c for i, c in enumerate(wall_contacts) if abs(c - wall_contacts[i - 1]) < 2]

    impact_indices = list(wall_contacts[0:1]) + [c for i, c in enumerate(wall_contacts)
                                                 if c - wall_contacts[i - 1] > int(x.fps * 2)]
    return impact_indices, contact


def reduce_contact_points(contact):
    # only the contact points that are far enough away from each other.
    if len(contact) == 0:
        return []
    else:
        a = np.array(contact)
        a = a[a[:, 1].argsort()]
        contact_points = [[a[0]]]

        for a0, a1 in zip(a[:-1], a[1:]):
            if a0[1] - a1[1] > 1:
                contact_points.append([a1])
            else:
                contact_points[-1].append(a1)

        contact_points = [np.array(c).mean(axis=0) for c in np.array(contact_points)]
        return contact_points


def Contact_analyzer(x, *args, **kwargs):
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
    impact_indices, contact = find_Impact_Points(x, my_maze, *args, *kwargs)

    delta_t = 2
    delta_frames = x.fps * delta_t

    theta_dot, torque = [], []
    for impact_index in impact_indices:
        start_frame = max(0, impact_index - delta_frames)
        end_frame = [i for i in range(impact_index, min(len(contact) - 1, impact_index + delta_frames))
                     if len(contact[i]) > 0][-1]

        v0 = np.mean(velocity_x(x, 1, 'x', 'y')[:, start_frame:impact_index], axis=1)

        torque_i, force_parallel, rhos = [], [], []
        contact[impact_index:end_frame] = [reduce_contact_points(c) for c in contact[impact_index:end_frame]]

        for c in contact[impact_index]:
            rhos.append(x.position[impact_index] - c)
            torque_i.append(np.cross(rhos[-1], v0))
            # force_parallel.append(np.dot(v0, rhos[-1])/np.linalg.norm(rhos[-1]))

        torque.append(np.sum(torque_i))

        # Characterize rotation
        r_impact = x.position[impact_index] - contact[impact_index][0]
        r_end = x.position[end_frame] - contact[end_frame][0]

        theta_dot.append((theta(r_end) - theta(r_impact)) / delta_t)

        # rho_dot = (np.linalg.norm(r_end) - np.linalg.norm(r_0))/delta_t

        # I want to flip the ones contacting the bottom corner...
        if np.mean(np.array(contact[impact_index])[:, 1]) < my_maze.arena_height / 2 - my_maze.exit_size / 2 + 0.1:
            theta_dot[-1], torque[-1], = -theta_dot[-1], -torque[-1],

        print('\ntheta_dot = ' + str(theta_dot[-1]))
        print('torque = ' + str(torque[-1]))

        if torque[-1] * theta_dot[-1] < - 0.05:
            x.play(1, 'contact', 'Display',
                   indices=[start_frame, end_frame],
                   wait=60,
                   arrows=[(x.position[impact_index],
                            x.position[impact_index] + v0 / (0.5 * np.linalg.norm(v0)), 'v') for fr in x.frames])
            print()

    information = [x.filename + '\n ' + str(impact_index) for impact_index in impact_indices]
    return theta_dot, torque, information


def Contact_analyzer_2(x, *args, **kwargs):
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
    impact_indices, contact = find_Impact_Points(x, my_maze, *args, *kwargs)

    delta_t = 2
    delta_frames = x.fps * delta_t

    theta_dot, torque = [], []
    for impact_index in impact_indices:
        start_frame = max(0, impact_index - delta_frames)
        end_frame = [i for i in range(impact_index, min(len(contact) - 1, impact_index + delta_frames))
                     if len(contact[i]) > 0][-1]
        contact[impact_index:end_frame] = [reduce_contact_points(c) for c in contact[impact_index:end_frame]]

        # allowed and prohibited directions
        torque_i, rhos, arrows, rho_cross = [], [], [], []
        for c in contact[impact_index]:
            rhos.append(x.position[impact_index] - c)
            rho_cross.append(np.cross(np.hstack([rhos[-1], 1]), [0, 0, 1]))
            arrows.append([(x.position[impact_index], x.position[impact_index] + rhos[-1], 'prohibited')
                           # , (x.position[impact_index], x.position[impact_index] + rho_cross[-1][:2], 'allowed')
                           for fr in x.frames])  # /(0.5*np.linalg.norm(rhos[-1]))

        # # torque and initial velocity
        # torque_i, rhos = [], []
        # v0 = np.mean(velocity_x(x, 1, 'x', 'y')[:, start_frame:impact_index], axis=1)
        # for c in contact[impact_index]:
        #     rhos.append(x.position[impact_index] - c)
        #     torque_i.append(np.cross(rhos[-1], v0))
        # torque.append(np.sum(torque_i))

        # Characterize rotation
        # r_impact = x.position[impact_index] - contact[impact_index][0]
        # r_end = x.position[end_frame] - contact[end_frame][0]
        #
        # theta_dot.append((theta(r_end) - theta(r_impact)) / delta_t)
        #
        # # I want to flip the ones contacting the bottom corner...
        # if np.mean(np.array(contact[impact_index])[:, 1]) < my_maze.arena_height / 2 - my_maze.exit_size / 2 + 0.1:
        #     theta_dot[-1], torque[-1], = -theta_dot[-1], -torque[-1],
        #
        # print('\ntheta_dot = ' + str(theta_dot[-1]))
        # print('torque = ' + str(torque[-1]))

        # if torque[-1] * theta_dot[-1] < - 0.05:
        x.play(1, 'contact', 'Display',
               indices=[start_frame, end_frame],
               wait=60,
               arrows=arrows)
        print()

    information = [x.filename + '\n ' + str(impact_index) for impact_index in impact_indices]
    return theta_dot, torque, information
