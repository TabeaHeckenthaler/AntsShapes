# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:55:29 2020
@author: tabea
"""
import numpy as np

from Directories import NewFileName
from PhysicsEngine.Gillespie import Gillespie

time_step = 0.01


def step(my_load, x, my_maze, pause, display=None, **kwargs):
    gillespie = kwargs['gillespie']

    if gillespie.time_until_next_event < time_step and not pause:
        gillespie.time_until_next_event = gillespie.whatsNext(my_load)

    forces(my_load, pause=pause, display=display, **kwargs)

    if not pause:
        gillespie.time_until_next_event -= time_step
        my_maze.Step(time_step, 10, 10)

        x.position = np.vstack((x.position, [my_load.position.x, my_load.position.y]))
        x.angle = np.hstack((x.angle, my_load.angle))

    return


def forces(my_load, pause=False, display=None, **kwargs):
    my_load.linearVelocity = 0 * my_load.linearVelocity
    my_load.angularVelocity = 0 * my_load.angularVelocity

    gillespie = kwargs['gillespie']

    """ Magnitude of forces """
    display.arrows = []

    for i in range(len(gillespie.n_p)):
        start = gillespie.attachment_site_world_coord(my_load, i)
        end = None

        if gillespie.n_p[i]:
            f_x, f_y = gillespie.ant_force(my_load, i, pause=pause)

            start = start
            end = start + [10 * f_x, 10 * f_y]
            display.arrows.append((start, end, 'puller'))

        elif gillespie.n_l[i]:
            display.arrows.append((start, end, 'lifter'))

        else:
            display.arrows.append((start, end, 'empty'))


def mazeSimulation(size, shape, frames, init_angle=np.array([0.0]), display=True, free=False):
    # init_angle = np.array([rd.uniform(0, 1) * (2 * np.pi)])
    from PhysicsEngine.mainGame import mainGame
    from trajectory_inheritance.trajectory import Trajectory
    from Setup.Maze import Maze
    from Setup.Load import Load
    """
    Instantiate a x-Object
    """
    x = Trajectory(size=size, shape=shape, solver='sim',
                   filename=NewFileName('', size, shape, expORsim='sim'))
    """
    Here are all the parameters: 
    """
    x.frames = np.linspace(1, frames, frames)
    x.contact = [[] for _ in range(frames)]

    """
    Now start instantiating the world and the load... 
    """
    my_maze = Maze(size=size, shape=shape, solver='sim', free=free)
    my_load = Load(my_maze)

    x.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
    x.angle = init_angle  # array to store the position and angle of the load

    gillespie = Gillespie(my_load, x=x)
    wait = 0  # number of ms the system waits between every step. This slows down the simulation!
    x = mainGame(x, display=display, gillespie=gillespie, free=free, wait=wait)
    return x


if __name__ == '__main__':
    frames = 6000
    # possible shapes: 'I', 'circle', 'SPT' (here, Gillespie is implemented)
    # free is either False or True
    # size only relevant for 'I': (XS, S, M, SL, L, XL) and 'SPT' (S, M, L, XL)

    shape = 'I'
    my_trajectory = mazeSimulation(size='L', shape='SPT', frames=frames, display=True, free=False)

    # Save(my_trajectory)
    # my_trajectory.play()


