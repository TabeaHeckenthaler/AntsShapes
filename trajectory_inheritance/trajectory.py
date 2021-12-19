
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
import numpy as np
from os import path
import pickle
from Directories import SaverDirectories, work_dir, mini_SaverDirectories
from copy import deepcopy
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from scipy.signal import savgol_filter
from Analysis.Velocity import velocity

""" Making Directory Structure """
shapes = {'ant': ['SPT', 'H', 'I', 'T', 'RASH', 'LASH'],
          'human': ['SPT']}
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['Small Far', 'Small Near', 'Medium', 'Large'],
         'humanhand': ''}
solvers = ['ant', 'human', 'humanhand', 'ps_simulation']


length_unit = {'ant': 'cm', 'human': 'm',  'humanhand': 'cm', 'ps_simulation': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


def get(filename):
    import os
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            if filename in os.listdir(work_dir+dir):
                address = work_dir + os.path.join(dir, filename)
                with open(address, 'rb') as f:
                    x = pickle.load(f)
                return x
    else:
        raise ValueError('I cannot find ' + filename)


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

    def step(self, my_maze, i, display=None):
        my_maze.set_configuration(self.position[i], self.angle[i])

    def smooth(self):
        self.position[:, 0] = savgol_filter(self.position[:, 0], self.fps+1, 3)
        self.position[:, 1] = savgol_filter(self.position[:, 1], self.fps+1, 3)
        self.angle = savgol_filter(np.unwrap(self.angle), self.fps+1, 3) % (2 * np.pi)

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

    def iterate_coords(self) -> iter:
        for pos, angle in zip(self.position, self.angle):
            yield pos[0], pos[1], angle

    def find_contact(self):
        from PhysicsEngine.Contact import contact_loop_experiment
        my_maze = Maze(self)
        my_load = my_maze.bodies[-1]
        contact = []

        i = 0
        while i < len(self.frames):
            self.step(my_maze, i)  # update_screen the position of the load (very simple function, take a look)
            contact.append(contact_loop_experiment(my_load, my_maze))
            i += 1
        return contact

    def velocity(self, second_smooth, *args):
        return velocity(self.position, self.angle, self.fps, self.size, self.shape, second_smooth, self.solver, *args)

    def play(self, indices=None, wait=0, ps=None, step=1, videowriter=False):
        """
        Displays a given trajectory_inheritance (self)
        :param indices: which slice of frames would you like to display
        :param wait: how many milliseconds should we wait between displaying steps
        :param ps: Configuration space
        :param step: display only the ith frame
        :Keyword Arguments:
            * *indices* (``[int, int]``) --
              starting and ending frame of trajectory_inheritance, which you would like to display
        """
        x = deepcopy(self)

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if indices is None:
            indices = [0, -2]

        f1, f2 = int(indices[0]), int(indices[1]) + 1
        x.position, x.angle = x.position[f1:f2:step, :], x.angle[f1:f2:step]
        x.frames = x.frames[f1:f2:step]

        if hasattr(x, 'participants') and x.participants is not None:   # TODO: this is a bit ugly, why does Amirs
            # have participants?
            x.participants.positions = x.participants.positions[f1:f2:step, :]
            x.participants.angles = x.participants.angles[f1:f2:step]
            if hasattr(x.participants, 'forces'):
                x.participants.forces.abs_values = x.participants.forces.abs_values[f1:f2:step, :]
                x.participants.forces.angles = x.participants.forces.angles[f1:f2:step, :]

        my_maze = Maze(x)
        return x.run_trj(my_maze, display=Display(x, my_maze, wait=wait, ps=ps, videowriter=videowriter))

    def save(self, address=None) -> None:
        """
        1. save a pickle of the object
        2. save a pickle of a tuple of attributes of the object, in case I make a mistake one day, and change attributes
        in the class and then am incapable of unpickling my files.
        """
        if address is None:
            address = SaverDirectories[self.solver] + path.sep + self.filename

        with open(address, 'wb') as f:
            try:
                self_copy = deepcopy(self)
                if hasattr(self_copy, 'participants'):
                    delattr(self_copy, 'participants')
                pickle.dump(self_copy, f)
                print('Saving ' + self_copy.filename + ' in ' + address)
            except pickle.PicklingError as e:
                print(e)

        print('Saving minimal' + self.filename + ' in path: ' + mini_SaverDirectories[self.solver])
        pickle.dump((self.shape, self.size, self.solver, self.filename, self.fps,
                     self.position, self.angle, self.frames, self.winner),
                    open(mini_SaverDirectories[self.solver] + path.sep + self.filename, 'wb'))

    def load_participants(self):
        pass

    def averageCarrierNumber(self):
        pass

    def run_trj(self, my_maze, display=None):
        i = 0
        while i < len(self.frames) - 1:
            self.step(my_maze, i, display=display)
            i += 1
            if display is not None:
                end = display.update_screen(self, i)
                if end:
                    display.end_screen()
                    self.frames = self.frames[:i]
                    break
                display.renew_screen(frame=self.frames[i], movie_name=self.filename)
        if display is not None:
            display.end_screen()

    def initial_cond(self):
        """
        We changed the initial condition. First, we had the SPT start between the two slits.
        Later we made it start in the back of the room.
        :return: str 'back' or 'front' depending on where the shape started
        """
        if self.shape != 'SPT':
            return None
        elif self.position[0, 0] < Maze(self).slits[0]:
            return 'back'
        return 'front'

    def communication(self):
        return False


