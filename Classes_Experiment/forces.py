import numpy as np
from Analysis_Functions.Velocity import crappy_velocity
from trajectory import Get
from Classes_Experiment.humans import Humans
from Setup.Load import force_attachment_positions

angle_shift = {0: 0,
               1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
               4: np.pi, 5: np.pi,
               6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2}
force_scaling_factor_DISPLAY = 1 / 5


def force_in_frame(x, i):
    frame = x.participants.frames[i]
    return [[frame.forces[name] * norm_force_vector(x, i, name)] for name in x.participants.occupied]


def norm_force_vector(x, i, name):
    angle = x.angle[i] + x.participants.frames[i].angle[name] + angle_shift[name]
    return np.array([np.cos(angle), np.sin(angle)])


def participants_force_arrows(x, my_load, i):
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
    x = Get('medium_20201221135753_20201221140218', 'human')
    x.participants = Humans(x)
    x.play(forces=[participants_force_arrows])
