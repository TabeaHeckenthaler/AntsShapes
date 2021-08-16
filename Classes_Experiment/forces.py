from Box2D import b2Vec2
import numpy as np
from Analysis_Functions.Velocity import crappy_velocity

angle_shift = {0: 0,
               1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
               4: np.pi, 5: np.pi,
               6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2}
force_scaling_factor = 1 / 5


def force_in_frame(x, i):
    frame = x.participants.frames[i]
    return [[frame.forces[name] * norm_force_vector(x, i, name)] for name in x.participants.occupied]


def norm_force_vector(x, i, name):
    angle = x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name]
    return np.array([np.cos(angle), np.sin(angle)])


def force_attachment_positions(my_load, x):
    from Classes_Experiment.humans import participant_number
    from Setup.Load import getLoadDim, shift
    if x.solver == 'human' and x.size == 'Medium' and x.shape == 'SPT':

        # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)
        a29, a38, a47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4

        # (0, 0) is the middle of the shape
        positions = [[shape_width / 2, 0],
                     [a29, shape_thickness / 2],
                     [a38, shape_thickness / 2],
                     [a47, shape_thickness / 2],
                     [-shape_width / 2, shape_height / 4],
                     [-shape_width / 2, -shape_height / 4],
                     [a47, -shape_thickness / 2],
                     [a38, -shape_thickness / 2],
                     [a29, -shape_thickness / 2]]

        # shift the shape...
        h = shift * shape_width
        positions = [[r[0] - h, r[1]] for r in positions]

    elif x.solver == 'human' and x.size == 'Large' and x.shape == 'SPT':
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)

        xMNOP = -shape_width/2,
        xLQ = xMNOP + shape_thickness / 2
        xAB = (-1) * xMNOP
        xCZ = (-1) * xLQ
        xKR = xMNOP + shape_thickness
        xDY, xEX, xFW, xGV, xHU, xIT, xJS = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

        yA_B = short_edge/6
        yC_Z = short_edge/2
        yDEFGHIJ_STUVWXY = shape_thickness/2
        yK_R = shape_height/10 * 2
        yL_Q = shape_height/2
        yM_P =  shape_height/10 * 3
        yN_O = shape_height/10

        positions = [[xAB, yA_B],
                     [xAB, - yA_B],
                     [xCZ, yC_Z],
                     [xDY, yDEFGHIJ_STUVWXY],
                     [xEX, yDEFGHIJ_STUVWXY],
                     [xFW, yDEFGHIJ_STUVWXY],
                     [xGV, yDEFGHIJ_STUVWXY],
                     [xHU, yDEFGHIJ_STUVWXY],
                     [xIT, yDEFGHIJ_STUVWXY],
                     [xJS, yDEFGHIJ_STUVWXY],
                     [xKR, yK_R],
                     [xLQ, yL_Q],
                     [xMNOP, yM_P],
                     [xMNOP, yN_O],
                     [xMNOP, -yN_O],
                     [xMNOP, -yM_P],
                     [xLQ, -yL_Q],
                     [xKR, -yK_R],
                     [xJS, -yDEFGHIJ_STUVWXY],
                     [xIT, -yDEFGHIJ_STUVWXY],
                     [xHU, -yDEFGHIJ_STUVWXY],
                     [xGV, -yDEFGHIJ_STUVWXY],
                     [xFW, -yDEFGHIJ_STUVWXY],
                     [xEX, -yDEFGHIJ_STUVWXY],
                     [xDY, -yDEFGHIJ_STUVWXY],
                     [xCZ, -yC_Z],
                     ]
        #
        # # shift the shape...
        # h = shift * shape_width
    else:
        positions = [[0, 0] for i in range(participant_number[x.size])]
    return [my_load.GetWorldPoint(b2Vec2(r)) for r in positions]


def participants_force_arrows(x, my_load, i):
    arrows = []
    frame = x.participants.frames[i]

    for name in x.participants.occupied:
        force = (frame.forces[name]) * force_scaling_factor
        force_meter_coor = force_attachment_positions(my_load, x)[name]
        if abs(force) > 0.2:
            arrows.append((force_meter_coor,
                           force_meter_coor + force * norm_force_vector(x, i, name),
                           str(name + 1)))
        # if abs(x.participants.frames[i].angle[name]) > np.pi / 2:
        #     print()
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