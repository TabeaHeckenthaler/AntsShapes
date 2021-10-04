import numpy as np
from Analysis_Functions.Velocity import crappy_velocity
from trajectory import Get
from Classes_Experiment.humans import Humans
from Setup.Load import force_attachment_positions

angle_shift = {'Medium': {0: 0,
                     1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
                     4: np.pi, 5: np.pi,
                     6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2},

               # for Large, this describes the number keys describe the force meter identities.
               'Large': {0: np.pi/2, 1: np.pi/2, 2: np.pi/2, 3: np.pi/2,
                     4: 0,
                     5: np.pi/2,
                     6: np.pi, 7: np.pi, 8: np.pi, 9: np.pi,
                     10: -np.pi/2,
                     11: 0,
                     12: -np.pi/2, 13: -np.pi/2, 14: -np.pi/2, 15: -np.pi/2, 16: -np.pi/2, 17: -np.pi/2,18: -np.pi/2,
                     19: -np.pi/2,
                     20: 0, 21: 0,
                     22: np.pi/2,
                     23: np.pi/2, 24: np.pi/2, 25: np.pi/2},
               }
force_scaling_factor_DISPLAY = 1 / 5


def force_in_frame(x, i):
    """
    param x: trajectory
    param i: frame number
    return 2x1 numpy array with x and y component of the normalized force vector
    """
    frame = x.participants.frames[i]
    return [[frame.forces[name] * norm_force_vector(x, i, name)] for name in x.participants.occupied]


def norm_force_vector(x, i, name):
    """
    param x: trajectory
    param i: frame number
    param name: integer, which describes the position of a participant, where A is index 0, ... and Z is index 26
    return 2x1 numpy array with x and y component of the normalized force vector
    """
    angle = x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[x.size][name]
    return np.array([np.cos(angle), np.sin(angle)])


def participants_force_arrows(x, my_load, i):
    arrows = []
    if len(x.participants.frames) == 0:
        raise Exception('Either you have no force measurement or you have not configured it. Check in Testable!')

    frame = x.participants.frames[i]
    for name in x.participants.occupied:
        # force = (frame.forces[name]) * force_scaling_factor_DISPLAY # TODO: change back
        force = 1
        force_meter_coor = force_attachment_positions(my_load, x)[name]
        if abs(force) > 0.2:
            arrows.append((force_meter_coor,
                           force_meter_coor + force * norm_force_vector(x, i, name),
                           str(name + 1)))  # name + 1 because A is 1 in Aviram's analysis, and in my list A is 0.
        # if abs(x.participants.frames[i].angle[name]) > np.pi / 2:
        #     print()
    return arrows


def center_pull_point_vectors(x, my_load, i):
    arrows = []
    if len(x.participants.frames) == 0:
        raise Exception('Either you have no force measurement or you have not configured it. Check in Testable!')

    frame = x.participants.frames[i]
    for name in x.participants.occupied:
        force = (frame.forces[name]) * force_scaling_factor_DISPLAY
        force_meter_coor = force_attachment_positions(my_load, x)[name]
        if abs(force) > 0.2:
            arrows.append((force_meter_coor,
                           force_meter_coor + force * norm_force_vector(x, i, name),
                           str(name + 1)))
        # if abs(x.participants.frames[i].angle[name]) > np.pi / 2:
        #     print()
    return arrows


def correlation_force_velocity(x, i):
    net_force = np.sum(np.array(force_in_frame(x, i)), axis=0)
    velocity = crappy_velocity(x, i)
    return np.vdot(net_force, velocity)


""" Look at single experiments"""
if __name__ == '__main__':
    x = Get('large_20210419100024_20210419100547', 'human')
    x.participants = Humans(x)
    x.play(forces=[participants_force_arrows])
