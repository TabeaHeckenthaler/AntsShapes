from Box2D import b2Vec2
import numpy as np
from Analysis_Functions.Velocity import crappy_velocity

angle_shift = {0: 0,
               1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
               4: np.pi, 5: np.pi,
               6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2}
force_scaling_factor = 1 / 5


def force_in_frame(x, i):
    return [[(x.participants.frames[i].forces[name]) / 5 * comp
             for comp in
             [np.cos(x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name]),
              np.sin(x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name])]]
            for name in x.participants.occupied]


def force_attachment_positions(my_load, x):
    from Classes_Experiment.humans import participant_number
    from Setup.Load import getLoadDim, shift
    if x.solver == 'human' and x.size == 'Medium' and x.shape == 'SPT':
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)
        a29, a38, a47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4

        positions = [[shape_width / 2, 0], [a29, shape_thickness / 2], [a38, shape_thickness / 2],
                     [a47, shape_thickness / 2], [-shape_width / 2, shape_height / 4],
                     [-shape_width / 2, -shape_height / 4],
                     [a47, -shape_thickness / 2], [a38, -shape_thickness / 2], [a29, -shape_thickness / 2]]

        # shift the shape...
        h = shift * shape_width
        positions = [[r[0] - h, r[1]] for r in positions]

    elif x.solver == 'human' and x.size == 'Large' and x.shape == 'SPT':
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)
        # a29, a38, a47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4
        #
        # positions = [[shape_width / 2, 0], [a29, shape_thickness / 2], [a38, shape_thickness / 2],
        #              [a47, shape_thickness / 2], [-shape_width / 2, shape_height / 4],
        #              [-shape_width / 2, -shape_height / 4],
        #              [a47, -shape_thickness / 2], [a38, -shape_thickness / 2], [a29, -shape_thickness / 2]]
        #
        # # shift the shape...
        # h = shift * shape_width
        positions = [[0, 0] for i in range(participant_number[x.size])]
    else:
        positions = [[0, 0] for i in range(participant_number[x.size])]
    return [my_load.GetWorldPoint(b2Vec2(r)) for r in positions]


def participants_force_arrows(x, my_load, i):
    arrows = []

    for name in x.participants.occupied:
        # x.participants.frames[i].angle[name] = 0
        # force = 1
        force = (x.participants.frames[i].forces[name]) * force_scaling_factor

        if abs(force) > 0.2:
            arrows.append((force_attachment_positions(my_load, x)[name],
                           force_attachment_positions(my_load, x)[name] +
                           [force * comp
                            for comp in [np.cos(x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name]),
                                         np.sin(
                                             x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name])]],
                           str(name + 1)))
        if abs(x.participants.frames[i].angle[name]) > np.pi / 2:
            print()
    return arrows


def net_force_arrows(x, my_load, i):
    if hasattr(x.participants.frames[i], 'forces'):
        start = x.position[i]
        end = x.position[i] + np.sum(np.array(force_in_frame(x, i)), axis=0)
        string = 'net force'
        return [(start, end, string)]
    else:
        return []


def correlation_force_velocity(x, my_load, i):
    net_force = np.sum(np.array(force_in_frame(x, i)), axis=0)
    velocity = crappy_velocity(x, i)
    return np.vdot(net_force, velocity)


""" Look at single experiments"""
# x = Get('human', 'medium_20201221135753_20201221140218')
# x.participants = Humans(x)
# x.play(1, 'Display', 'contact', forces=[participants_force_arrows])