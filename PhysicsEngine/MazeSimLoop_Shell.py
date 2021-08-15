# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:55:29 2020

@author: tabea
"""
import numpy as np
import random as rd  # import (gauss, uniform)
from trajectory import NewFileName, Trajectory
from Setup.Load import Loops
from PhysicsEngine.Display_Pygame import arrow


def step(my_load, x, my_maze, i, pause, **kwargs):
    TARGET_FPS = 100  # Frames per second
    TIME_STEP = 1 / TARGET_FPS  # time step in simulation (seconds)
    if not pause:
        ForceAttachments, arrows = Forces(my_load, x, my_maze, **kwargs)
        for a_i in arrows:
            arrow(*a_i)
        my_maze.Step(TIME_STEP, 10, 10)
        x.position = np.vstack((x.position, [my_load.position.x, my_load.position.y]))
        x.angle = np.hstack((x.angle, my_load.angle))
    return my_load, x, my_maze, i


def Forces(my_load, x, my_maze, **kwargs):
    from Setup.MazeFunctions import ClosestCorner
    from Box2D import b2Vec2
    grC = 1
    gravCenter = np.array([my_maze.arena_length * grC,
                           my_maze.arena_height / 2])  # this is the 'far away point', to which the load gravitates

    load_vertices = Loops(my_load)

    """ Where Force attaches """
    ForceAttachments = [ClosestCorner(load_vertices, gravCenter)]

    """ Magnitude of forces """
    arrows = []
    for ForceAttachment in ForceAttachments:
        f_x = -rd.gauss(x.xForce * (ForceAttachment[0] - my_maze.arena_length * grC) / my_maze.arena_length * grC,
                        x.xDev)
        f_y = -rd.gauss(x.yForce * (my_load.position.y - my_maze.arena_height / 2) / my_maze.arena_height / 2, x.yDev)

        f_x = 1
        f_y = 0

        my_load.ApplyForce(b2Vec2([f_x, f_y]),
                           b2Vec2(ForceAttachment),
                           True)

        start = ForceAttachment
        end = ForceAttachment + [f_x, f_y]
        arrows.append((start, end, ''))

    return ForceAttachments, arrows


def MazeSimulation(size, shape, frames, init_angle=np.array([rd.uniform(0, 1) * (2 * np.pi)]), **kwargs):
    from PhysicsEngine.Box2D_GameLoops import MainGameLoop
    from trajectory import Trajectory
    from Setup.Maze import Maze
    """
    Instatiate a x-Object
    """
    solver='sim'
    x = Trajectory(size=size, shape=shape, solver=solver, filename=NewFileName('', size, shape, solver))
    """
    Here are all the parameters: 
    """
    x.buffer = 0
    x.xForce, x.xDev, x.yForce, x.yDev = 1, 10, 0, 5  # These numbers give a magnitude to the force acting towards

    # the gravitational center at distance grC*MazeLength
    x.linearDamping, x.angularDamping = 0.1, 0.1  # Damping coefficient
    x.friction, x.restitution = 0, 0.5
    x.frames = np.linspace(1, frames, frames)
    x.contact = [[] for _ in range(frames)]

    """
    Now start instantiating the World and the load... 
    """
    my_maze = Maze(size=size, shape=shape, solver=solver)
    if '_sim_' in x.filename:
        x.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
        x.angle = init_angle  # array to store the position and angle of the load

    x = MainGameLoop(x, 'Display')
    return x

#
# frames = 10000
# MazeSimulation('XL', 'H', frames)
