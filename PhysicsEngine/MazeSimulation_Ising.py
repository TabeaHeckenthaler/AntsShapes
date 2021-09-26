# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:55:29 2020
@author: tabea
"""
import random as rd  # import (gauss, uniform)

import numpy as np
from Box2D import b2Vec2

from Setup.Load import Loops
from Setup.MazeFunctions import ClosestCorner
from trajectory import NewFileName, Save
from PhysicsEngine.Gillespie import Gillespie

# TARGET_FPS = 100  # Frames per second
# TIME_STEP = 1 / TARGET_FPS  # time step in simulation (seconds)


def step(my_load, x, my_maze, pause, **kwargs):
    arrows = None
    if not pause:
        TIME_STEP = kwargs['gillespie'].whatsNext(my_load)
        ForceAttachments, arrows = Forces(my_load, my_maze, **kwargs)
        my_maze.Step(TIME_STEP, 10, 10)

        x.position = np.vstack((x.position, [my_load.position.x, my_load.position.y]))
        x.angle = np.hstack((x.angle, my_load.angle))
    return arrows


def Forces_old(my_load, my_maze, **kwargs):
    grC = 1
    gravCenter = np.array([my_maze.arena_length * grC, my_maze.arena_height / 2])  # this is the 'far away point', to which the load gravitates

    load_vertices = Loops(my_load)

    """ Where Force attaches """
    ForceAttachments = [ClosestCorner(load_vertices, gravCenter)]

    """ Magnitude of forces """
    arrows = []
    for ForceAttachment in ForceAttachments:
        # f_x = -rd.gauss(x.xForce * (ForceAttachment[0] - my_maze.arena_length * grC) / my_maze.arena_length * grC,
        #                 x.xDev)
        # f_y = -rd.gauss(x.yForce * (my_load.position.y - my_maze.arena_height / 2) / my_maze.arena_height / 2, x.yDev)

        f_x = 1
        f_y = 0

        my_load.ApplyForce(b2Vec2([f_x, f_y]),
                           # b2Vec2(ForceAttachment),
                           my_load.position,
                           True)

        start = ForceAttachment
        end = ForceAttachment + [f_x, f_y]
        arrows.append((start, end, ''))

    return ForceAttachments, arrows


def Forces(my_load, my_maze, **kwargs):
    my_load.linearVelocity = 0 * my_load.linearVelocity
    my_load.angularVelocity = 0 * my_load.angularVelocity

    ForceAttachments = list()  # in which coordinate system is this?
    gillespie = kwargs['gillespie']

    """ Magnitude of forces """
    arrows = []

    for i in np.where(gillespie.n_p)[0]:
        f_x, f_y = gillespie.ant_force(my_load, i)
        ForceAttachments.append(gillespie.attachment_position(my_load, i))

        start = ForceAttachments[-1]
        end = ForceAttachments[-1] + [1000 * f_x, 1000 * f_y]
        arrows.append((start, end, 'puller'))

    for i in np.where(gillespie.n_l)[0]:
        ForceAttachments.append(gillespie.attachment_position(my_load, i))

        start = ForceAttachments[-1]
        end = None
        arrows.append((start, end, 'lifter'))
    return ForceAttachments, arrows


def MazeSimulation(size, shape, frames, init_angle=np.array([0.0]), display=True, free=False):
    # init_angle = np.array([rd.uniform(0, 1) * (2 * np.pi)])
    from PhysicsEngine.Box2D_GameLoops import MainGameLoop
    from trajectory import Trajectory
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
    # x.xForce, x.xDev, x.yForce, x.yDev = 1, 10, 0, 5  # These numbers give a magnitude to the force acting towards
    #
    # # the gravitational center at distance grC*MazeLength
    # x.linearDamping, x.angularDamping = 0.1, 0.1  # Damping coefficient
    # x.friction, x.restitution = 0, 0.5
    x.frames = np.linspace(1, frames, frames)
    x.contact = [[] for _ in range(frames)]

    """
    Now start instantiating the world and the load... 
    """
    my_maze = Maze(size=size, shape=shape, solver='sim', free=free)
    my_load = Load(my_maze)

    x.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
    x.angle = init_angle  # array to store the position and angle of the load

    gillespie = Gillespie()
    gillespie.new_attachment(0, my_load, type='puller')
    gillespie.new_attachment(5, my_load, type='puller')
    gillespie.phi[5] = 1
    gillespie.new_attachment(10, my_load, type='lifter')
    x = MainGameLoop(x, display=display, gillespie=gillespie, free=free)
    return x


if __name__ == '__main__':
    frames = 6000
    # my_trajectory = MazeSimulation(size='XL', shape='H', frames=frames, display=True)
    my_trajectory = MazeSimulation(size='', shape='circle', frames=frames,
                                   display=True, free=True)

    # Save(my_trajectory)
    # my_trajectory.play()


