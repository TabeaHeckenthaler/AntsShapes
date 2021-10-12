
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
import numpy as np
from os import path
import pickle
from PhysicsEngine.mainGame import mainGame
from Directories import SaverDirectories
from copy import deepcopy

""" Making Directory Structure """
shapes = {'ant': ['SPT', 'H', 'I', 'T', 'RASH', 'LASH'],
          'human': ['SPT']}
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['S', 'M', 'L'],
         'humanhand': ''}
solvers = ['ant', 'human', 'humanhand', 'ps_simulation']


def get(filename, solver, address=None):
    if solver == 'ant':
        from trajectory_inheritance.trajectory_ant import ant_address
        address = ant_address(filename, solver)

    if path.isfile(SaverDirectories[solver] + path.sep + filename):
        address = SaverDirectories[solver] + path.sep + filename
    else:
        print('cannot find file!')
    with open(address, 'rb') as f:
        x = pickle.load(f)
    return x


class Trajectory:
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool):
        self.shape = shape  # shape (maybe this will become name of the maze...) (H, I, T, SPT)
        self.size = size  # size (XL, SL, L, M, S, XS)
        self.solver = solver  # ant, human, sim, humanhand
        self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        self.VideoChain = [self.filename]
        self.fps = fps  # frames per second
        self.position = np.empty((1, 2), float)  # np.array of x and y positions of the centroid of the shape
        self.angle = np.empty((1, 1), float)  # np.array of angles while the shape is moving
        self.frames = np.empty(0, float)
        self.winner = winner  # whether the shape crossed the exit
        self.participants = None

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def step(self, *args):
        pass

    def interpolate_over_NaN(self):
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            nan_frames = np.unique(np.append(np.where(np.isnan(self.position))[0], np.where(np.isnan(self.angle))[0]))

            fr = [[nan_frames[0]]]
            for i in range(len(nan_frames) - 1):
                if abs(nan_frames[i] - nan_frames[i + 1]) > 1:
                    fr[-1] = fr[-1] + [nan_frames[i]]
                    fr = fr + [[nan_frames[i + 1]]]
            fr[-1] = fr[-1] + [nan_frames[-1]]
            print('Was NaN...' + str([self.frames[i].tolist() for i in fr]))

        # Some of the files contain NaN values, which mess up the Loading.. lets interpolate over them
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            for indices in fr:
                if indices[0] < 1:
                    indices[0] = 1
                if indices[1] > self.position.shape[0] - 2:
                    indices[1] = indices[1] - 1
                con_frames = indices[1] - indices[0] + 2
                self.position[indices[0] - 1: indices[1] + 1, :] = np.transpose(np.array(
                    [np.linspace(self.position[indices[0] - 1][0], self.position[indices[1] + 1][0], num=con_frames),
                     np.linspace(self.position[indices[0] - 1][1], self.position[indices[1] + 1][1], num=con_frames)]))
                self.angle[indices[0] - 1: indices[1] + 1] = np.squeeze(np.transpose(
                    np.array([np.linspace(self.angle[indices[0] - 1], self.angle[indices[1] + 1], num=con_frames)])))

    def timer(self):
        return (len(self.frames) - 1) / self.fps

    def play(self, interval=1, PhaseSpace=None, ps_figure=None, wait=0, indices=None, **kwargs):
        r"""Displays a given trajectory_inheritance (self)

        :Non-key-worded Arguments:
            * *attempt* --
              when passed, an attempt zone is installed and a parameter

        :Keyword Arguments:
            * *interval* (``int``) --
              Display only every nth frame
            * *PhaseSpace* (``PhaseSpace``) --
              PhaseSpace in which the shape is moving
            * *ps_figure* (``mayavi figure``) --
              figure in which the PhaseSpace is displayed and the trajectory_inheritance shall be drawn
            * *wait* (``int``) --
              milliseconds between the display of consecutive frames
            * *indices* (``[int, int]``) --
              starting and ending frame of trajectory_inheritance, which you would like to display
            * *attempt* (``bool``) --
              milliseconds between the display of consecutive frames
        """
        x = deepcopy(self)

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if indices is not None:
            f1, f2 = int(indices[0]), int(indices[1]) + 1
            x.position, x.angle = x.position[f1:f2, :], x.angle[f1:f2]
            x.frames = x.frames[int(f1):int(f2)]

        return mainGame(x, display=True, interval=interval, PhaseSpace=PhaseSpace, ps_figure=ps_figure, wait=wait,
                        **kwargs)

    def save(self, address=None):
        if address is None:
            address = SaverDirectories[self.solver] + path.sep + self.filename

        with open(address, 'wb') as f:
            try:
                self_copy = deepcopy(self)
                delattr(self_copy, 'participants')
                pickle.dump(self_copy, f)
                print('Saving ' + self_copy.filename + ' in ' + address)
            except pickle.PicklingError as e:
                print(e)
        return

    def load_participants(self):
        pass

    def averageCarrierNumber(self):
        pass