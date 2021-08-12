# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
from scipy.spatial import cKDTree
import numpy as np
import glob
from Box2D import b2BodyDef
import matplotlib.pyplot as plt
import scipy.io as sio
from os import (listdir, getcwd, mkdir, path)
import pickle
import shutil
from copy import deepcopy
from Setup.MazeFunctions import BoxIt, PlotPolygon
from PhysicsEngine import Box2D_GameLoops

# import handheld

""" Making Directory Structure """
shapes = {'ant': ['SPT', 'H', 'I', 'T', 'RASH', 'LASH'],
          'human': ['SPT']}
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['S', 'M', 'L'],
         'humanhand': ''}
solvers = ['ant', 'human', 'humanhand', 'dstar']

home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)
work_dir = data_home + 'Pickled_Trajectories\\'
AntSaverDirectory = work_dir + 'Ant_Trajectories'
HumanSaverDirectory = work_dir + 'Human_Trajectories'
HumanHandSaverDirectory = work_dir + 'HumanHand_Trajectories'
DstarSaverDirectory = work_dir + 'Dstar_Trajectories'
SaverDirectories = {'ant': AntSaverDirectory,
                    'human': HumanSaverDirectory,
                    'humanhand': HumanHandSaverDirectory,
                    'dstar': DstarSaverDirectory}

length_unit = {'ant': 'cm',
               'human': 'm',
               'humanhand': 'cm',
               'dstar': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


def communication(filename, solver):
    if solver != 'human':
        return False
    else:
        from Classes_Experiment.humans import excel_worksheet_index, get_sheet
        index = excel_worksheet_index(filename)
        return get_sheet().cell(row=index, column=5).value == 'C'


def maze_size(size):
    maze_s = {'Large': 'L',
              'Medium': 'M',
              'Small Far': 'S',
              'Small Near': 'S'}
    if size in maze_s.keys():
        return maze_s[size]
    else:
        return size


def time(x, condition):
    if condition == 'winner':
        return x.time
    if condition == 'all':
        if x.winner:
            return x.time
        else:
            return 60 * 40  # time in seconds after which the maze supposedly would have been solved?


def Directories():
    if not (path.isdir(AntSaverDirectory)):
        breakpoint()
        mkdir(AntSaverDirectory)
        mkdir(AntSaverDirectory + path.sep + 'OnceConnected')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion' + path.sep + 'OnceConnected')
    if not (path.isdir(HumanSaverDirectory)):
        mkdir(HumanSaverDirectory)
    if not (path.isdir(HumanHandSaverDirectory)):
        mkdir(HumanHandSaverDirectory)
    if not (path.isdir(DstarSaverDirectory)):
        mkdir(DstarSaverDirectory)
    return


def NewFileName(old_filename, size, shape, expORsim):
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        # findall(r'[\d.]+', 'TXL1_sim_255')[1] #this is a function able to read the last digit of the string
        filename = size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            filename = filename.replace(old_filename.split('_')[0], size + '_' + shape)
        else:
            filename = filename.replace(size + shape, size + '_' + shape)
    return filename


def Get(filename, solver, address=None):
    if address is None:
        if path.isfile(SaverDirectories[solver] + path.sep + filename):
            address = SaverDirectories[solver] + path.sep + filename

        elif path.isfile(SaverDirectories[solver] + path.sep + 'OnceConnected' + path.sep + filename):
            print('This must be an old file.... ')
            address = SaverDirectories[solver] + path.sep + 'OnceConnected' + path.sep + filename

        elif path.isfile(SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + filename):
            address = SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + filename

        elif path.isfile(
                SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected' + path.sep + filename):
            print('This must be an old file.... ')
            address = SaverDirectories[
                          solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected' + path.sep + filename
        else:
            print('I cannot find this file: ' + filename)
            return Trajectory()

    with open(address, 'rb') as f:
        x = pickle.load(f)
    if type(x.participants) == list:
        delattr(x, 'participants')
        Save(x)
    return x


def Save(x, address=None):
    Directories()
    if address is None:
        if x.solver in solvers:
            if x.free:
                address = SaverDirectories[x.solver] + path.sep + 'Free_Motion' + path.sep + x.filename
            else:
                address = SaverDirectories[x.solver] + path.sep + x.filename
        else:
            address = getcwd()

    with open(address, 'wb') as f:
        try:
            pickle.dump(x, f)
            print('Saving ' + x.filename + ' in ' + address)
        except pickle.PicklingError as e:
            print(e)
    # move_tail(x)
    return


def move_tail(x):
    if not x.free:
        origin_directory = SaverDirectories[x.solver]
        goal_directory = SaverDirectories[x.solver] + path.sep + 'OnceConnected'
    else:
        origin_directory = SaverDirectories[x.solver] + path.sep + 'Free_Motion'
        goal_directory = SaverDirectories[x.solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected'

    for tailFiles in x.VideoChain[1:]:
        if path.isfile(path.join(origin_directory, tailFiles)):
            shutil.move(path.join(origin_directory, tailFiles), goal_directory)


trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'


def MatlabFolder(solver, size, shape, free):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not free:
            return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
                shape] + path.sep + size + path.sep + 'Output Data'
        if free:
            return trackedAntMovieDirectory + path.sep + 'Free' + path.sep + 'Output Data' + path.sep + \
                   shape_folder_naming[shape]
    if solver == 'human':
        if not free:
            return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'
        if free:
            return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'
    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


class Trajectory:
    def __init__(self,
                 size=None, shape=None, solver=None,
                 filename=None,
                 free=False, fps=50,
                 winner=bool,
                 x_error=None, y_error=None, angle_error=None, falseTracking=None,
                 **kwargs):
        self.shape = shape  # shape (maybe this will become name of the maze...)
        self.size = size  # size
        self.solver = solver

        if 'old_filename' in kwargs:
            self.filename = NewFileName(kwargs['old_filename'], self.size, self.shape, 'exp')
        else:
            self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        self.free = free
        self.VideoChain = [self.filename]
        self.fps = fps  # frames per second

        self.position = np.empty((1, 2), float)  # np.array of x and y positions of the centroid of the shape
        self.angle = np.empty((1, 1), float)  # np.array of angles while the shape is moving
        self.frames = np.empty(0, float)
        self.tracked_frames = []

        if x_error is None:
            x_error = [0]
        if y_error is None:
            y_error = [0]
        if angle_error is None:
            angle_error = [0]
        if falseTracking is None:
            falseTracking = [[]]
        self.x_error, self.y_error, self.angle_error = x_error, y_error, angle_error
        self.falseTracking = falseTracking
        self.winner = winner  # whether the shape crossed the exit
        self.state = np.empty((1, 1), int)

    # def __copy__(self):
    #     return type(self)()

    def __add__(self, file2):
        max_distance_for_connecting = {'XS': 0.8, 'S': 0.2, 'M': 0.2, 'L': 0.2, 'SL': 0.2, 'XL': 0.2}
        from Setup.Load import periodicity
        if not (self.shape == file2.shape) or not (self.size == file2.size):
            print('It seems, that these files should not be joined together.... Please break... ')
            breakpoint()

        if abs(self.position[-1, 0] - file2.position[0, 0]) > max_distance_for_connecting[self.size] or \
                abs(self.position[-1, 1] - file2.position[0, 1]) > max_distance_for_connecting[self.size]:
            print('does not belong together')
            breakpoint()

        file12 = deepcopy(self)
        file12.x_error = self.x_error + file2.x_error  # these are lists that we want to join together
        file12.y_error = self.y_error + file2.y_error  # these are lists that we want to join together
        file12.angle_error = self.angle_error + file2.angle_error  # these are lists that we want to join together

        file12.position = np.vstack((self.position, file2.position))

        per = 2 * np.pi / periodicity[file12.shape]
        a0 = np.floor(self.angle[-1] / per) * per + np.mod(file2.angle[1], per)
        file12.angle = np.hstack((self.angle, file2.angle - file2.angle[1] + a0))
        file12.frames = np.hstack((self.frames, file2.frames))
        file12.tracked_frames = file12.tracked_frames + file2.tracked_frames

        if not self.free:
            # file12.contact = self.contact + file2.contact  # We are combining two lists here...
            file12.state = np.hstack((np.squeeze(self.state), np.squeeze(file2.state)))
            file12.winner = file2.winner  # The success of the attempt is determined, by the fact that the last file
            # is either winner or looser.

        file12.VideoChain = self.VideoChain + file2.VideoChain
        print(file12.VideoChain, sep="\n")

        file12.falseTracking = self.falseTracking + file2.falseTracking

        # Delete the load of filename
        return file12

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def old_filenames(self, i):
        if i >= len(self.VideoChain):
            return 'No video found (maybe because I extended)'

        if self.shape[1:] == 'ASH':
            if self.VideoChain[i].split('_')[0] + '_' + \
                    self.VideoChain[i].split('_')[1] == self.size + '_' + self.shape:
                old = self.VideoChain[i].replace(
                    self.VideoChain[i].split('_')[0] + '_' + self.VideoChain[i].split('_')[1],
                    self.size + self.shape[1:]) \
                      + '.mat'
            else:
                print('Something strange in x.old_filenames of x = ' + self.filename)
            #     # this is specifically for 'LASH_4160019_LargeLH_1_ants (part 1).mat'...
            #     old = self.VideoChain[i] + '.mat'
        else:
            old = self.VideoChain[i].replace(self.size + '_' + self.shape, self.size + self.shape) + '.mat'

            # if not('_sim_' in self.filename) and not(old in listdir(MatlabFolder(self.solver, self.size,
            # self.shape, self.free))) and 'CONNECTOR' not in self.VideoChain[i]:
        #     print('we didn't find the old filename')
        #     breakpoint()
        return old

    # Find the size and shape from the filename
    def shape_and_size(self, old_filename):
        if self.size == str('') and self.solver != 'humanhand':
            if len(old_filename.split('_')[0]) == 2:
                self.size = old_filename.split('_')[0][0:1]
                self.shape = old_filename.split('_')[0][1]
            if len(old_filename.split('_')[0]) == 3:
                self.size = old_filename.split('_')[0][0:2]
                self.shape = old_filename.split('_')[0][2]
            if len(old_filename.split('_')[0]) == 4:
                self.size = old_filename.split('_')[0][0:1]
                self.shape = old_filename.split('_')[0][1:4]  # currently this is only for size L and shape SPT
            if len(old_filename.split('_')[0]) == 5:
                self.size = old_filename.split('_')[0][0:2]
                self.shape = old_filename.split('_')[0][2:5]
        # now we figure out, what the zone is, if the arena size were equally scaled as the load and exit size.
        # arena_length, arena_height, x.exit_size, wallthick, slits, resize_factor = getMazeDim(x.shape, x.size)

    def matlab_loading(self, old_filename):
        if self.solver == 'ant':
            if not (old_filename == 'XLSPT_4280007_XLSpecialT_1_ants (part 3).mat'):
                file = sio.loadmat(
                    MatlabFolder(self.solver, self.size, self.shape, self.free) + path.sep + old_filename)

                if 'Direction' not in file.keys() and self.shape.endswith('ASH'):
                    # file['Direction'] = input('L2R or R2L  ')
                    file['Direction'] = None

                if self.shape.endswith('ASH') and 'R2L' == file['Direction']:
                    if self.shape == 'LASH':
                        self.shape = 'RASH'
                        self.filename.replace('LASH', 'RASH')
                        self.VideoChain = [name.replace('LASH', 'RASH') for name in self.VideoChain]

                    else:
                        self.shape = 'LASH'
                        self.filename.replace('RASH', 'LASH')
                        self.VideoChain = [name.replace('RASH', 'LASH') for name in self.VideoChain]

                if self.shape.endswith('ASH') and self.angle_error[0] == 0:
                    if self.shape == 'LASH':
                        self.angle_error = [2 * np.pi * 0.11 + self.angle_error[0]]
                    if self.shape == 'RASH':
                        self.angle_error = [-2 * np.pi * 0.11 + self.angle_error[
                            0]]  # # For all the Large Asymmetric Hs I had 0.1!!! (I think, this is why I needed the
                        # error in the end... )

                    if self.shape == 'LASH' and self.size == 'XL':  # # It seems like the exit walls are a bit
                        # crooked, which messes up the contact tracking
                        self.angle_error = [2 * np.pi * 0.115 + self.angle_error[0]]
                    if self.shape == 'RASH' and self.size == 'XL':
                        self.angle_error = [-2 * np.pi * 0.115 + self.angle_error[0]]

                load_center = file['load_center'][:, :]
                load_center[:, 0] = load_center[:, 0] + self.x_error
                load_center[:, 1] = load_center[:, 1] + self.y_error
                self.frames = file['frames'][0]
                self.tracked_frames = [file['frames'][0][0], file['frames'][0][-1]]
                # # Angle accounts for shifts in the angle of the shape.... (manually, by watching the movies)
                shape_orientation = \
                    np.matrix.transpose(file['shape_orientation'][:] * np.pi / 180 + self.angle_error[0])[
                        0]

            else:
                import h5py
                with h5py.File(MatlabFolder(self.solver, self.size, self.shape, self.free) + path.sep + old_filename,
                               'r') as f:
                    load_center = np.matrix.transpose(f['load_center'][:, :])
                    load_center[:, 0] = load_center[:, 0] + self.x_error
                    load_center[:, 1] = load_center[:, 1] + self.y_error
                    self.frames = np.matrix.transpose(f['frames'][:])[0]
                    # # Angle accounts for shifts in the angle of the shape.... (manually, by watching the movies)
                    shape_orientation = (f['shape_orientation'][:] * np.pi / 180 + self.angle_error[0])[0]

                    # # if not('Direction' in file.keys()) and not(self.shape == 'T' and self.size == 'S'):

        elif self.solver == 'human':
            file = sio.loadmat(MatlabFolder(self.solver, self.size, self.shape, self.free) + path.sep + old_filename)
            load_center = file['load_CoM'][:, 2:4]
            load_center[:, 0] = load_center[:, 0] + self.x_error
            load_center[:, 1] = load_center[:, 1] + self.y_error
            shape_orientation = np.matrix.transpose(file['orientation'][:] * np.pi / 180 + self.angle_error[0])[0]

            self.frames = np.linspace(1, load_center.shape[0], load_center.shape[0]).astype(int)

        elif self.solver == 'humanhand':
            humanhandPickle = pickle.load(open(MatlabFolder(self.solver, self.size, self.shape, self.free)
                                               + path.sep + self.filename + '.pkl', 'rb'))
            load_center = np.array(humanhandPickle.centers)
            shape_orientation = humanhandPickle.angles
            self.frames = humanhandPickle.frames

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)

        if self.solver == 'ant':
            from Analysis_Functions.Velocity import check_for_false_tracking
            self.falseTracking = [check_for_false_tracking(self)]
            self.falseTracker()

        # plt.plot(ConnectAngle(self.angle[2100 : 2300], 'LASH'), '*'); plt.show();
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
        return

    def participants(self):
        from Classes_Experiment.ants import Ants
        from Classes_Experiment.humans import Humans
        from Classes_Experiment.mr_dstar import Mr_dstar
        from Classes_Experiment.humanhand import Humanhand

        dc = {'ant': Ants,
              'human': Humans,
              'humanhand': Humanhand,
              'dstar': Mr_dstar
              }
        return dc[self.solver](self)

    def falseTracker(self):
        x = self
        from Setup.Load import periodicity
        per = periodicity[self.shape]

        for frames in x.falseTracking[0]:
            frame1, frame2 = max(frames[0] - 1, x.frames[0]), min(frames[1] + 1, x.frames[-1])
            index1, index2 = np.where(x.frames == frame1)[0][0], np.where(x.frames == frame2)[0][0]

            con_frames = index2 - index1
            x.position[index1: index2] = np.transpose(
                np.array([np.linspace(x.position[index1][0], x.position[index2][0], num=con_frames),
                          np.linspace(x.position[index1][1], x.position[index2][1], num=con_frames)]))

            # Do we cross either angle 0 or 2pi, when we connect through the shortest distance?
            if abs(x.angle[index2] - x.angle[index1]) / (2 * np.pi / per) > 0.7:
                x.angle[index2] = x.angle[index2] + np.round(
                    (x.angle[index1] - x.angle[index2]) / (2 * np.pi / per)) * (2 * np.pi / per)

            # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
            x.angle[index1: index2] = np.linspace(x.angle[index1], x.angle[index2], num=con_frames)
            for index in range(index1, index2):
                x.angle[index] = np.mod(x.angle[index], 2 * np.pi)

    def timer(self):
        return (len(self.frames) - 1) / self.fps

    def ZonedAngle(self, index, my_maze, angle_passed):
        angle_zoned = angle_passed
        if not (self.InsideZone(index, my_maze)) and self.position[-1][0] < self.zone[2][0]:
            angle_zoned = self.angle_zoned[index]
        return angle_zoned

    def ZonedPosition(self, index, my_maze):
        # first we check whether load is inside zone, then we check, whether the load has passed the first slit
        if self.position.size < 3:
            if not (self.InsideZone(index, my_maze)) and self.position[0] < self.zone[2][0]:
                zone = cKDTree(BoxIt(self.zone, 0.01))  # to read the data, type zone.data
                position_zoned = zone.data[zone.query(self.position)[1]]
            else:
                position_zoned = self.position
            if position_zoned[1] > 14.9:
                breakpoint()
        else:
            if not (self.InsideZone(index, my_maze)) and self.position[index][0] < self.zone[2][0]:
                zone = cKDTree(BoxIt(self.zone, 0.01))  # to read the data, type zone.data
                position_zoned = zone.data[zone.query(self.position[index])[1]]
            else:
                position_zoned = self.position[index]
            if position_zoned[1] > 14.9:
                breakpoint()
        return position_zoned

    def InsideZone(self, index, Maze):
        if self.position.size < 3:
            x = (Maze.zone[2][0] > self.position[0] > Maze.zone[0][0])
            y = (Maze.zone[1][1] > self.position[1] > Maze.zone[0][1])
        else:
            x = (Maze.zone[2][0] > self.position[index][0] > Maze.zone[0][0])
            y = (Maze.zone[1][1] > self.position[index][1] > Maze.zone[0][1])
        return x and y

    def first_nonZero_State(self):
        for i in range(1, len(self.state[1:] - 1)):
            if self.state[i] != 0:
                return i
        print('I did not find a non-Zero state')

    def first_Contact(self):
        contact = self.play(1, 'contact')[1]
        for i in range(1, len(self.state[1:] - 1)):
            if contact[i].size != 0 or self.state[i] != 0:
                return i

        print('I did not find a Contact state or a non-Zero state')

    # def last_Contact(self):
    #     for i in reversed(range(1,len(self.frames)-1)):
    #         if self.contact[i].size != 0 or self.state[i] != self.statenames[-1]:
    #             return i

    # print('Where is the last contact frame?')

    def plot(self, *vargs, **kwargs):
        from Setup.Maze import Maze
        from Setup.Load import AddLoadFixtures
        import matplotlib.cm as cm
        # here we plot the zone:
        plt.figure()

        ''' Plot the collision points on the shape & the point at which the load collides with the maze'''
        pos, con = np.array([0, 0]), np.array([0, 0])

        # if pos.size > 2:
        #     plt.plot(pos[:,0], pos[:,1], 'ro', markersize=3)
        #     plt.plot(con[:,0], con[:,1], 'ro', markersize=3)

        # Set the size of the figure... it should have the same ratio of the two axis so that it doenst look warped....
        fig = plt.gcf()
        ax = fig.gca()
        plt.grid()

        # Set the size of the figure... it should have the same ratio of the two axis so that it doenst look warped....
        fig.set_dpi(200)
        my_maze = Maze(size=self.size, shape=self.shape, solver=self.solver)
        if 'attempt' in vargs:
            from Setup.Attempts import AddAttemptZone
            my_maze, my_attempts_zone = AddAttemptZone(my_maze, self)

        ''' get the beginning and end position of load'''
        if 'frames' in kwargs:
            frames = kwargs['frames']
        else:
            frames = [0, -1]

        ''' Plot the path of the centroid of the load and if existent, the zoned path of the centroid '''
        plt.plot(self.position[frames[0]:frames[-1], 0], self.position[frames[0]:frames[-1], 1], 'b.',
                 markersize=3)  # draw the actual path

        for frame in frames:
            my_load_beginning = my_maze.CreateBody(
                b2BodyDef(position=(self.position[frame, 0], self.position[frame, 1]),
                          angle=self.angle[frame]))
            AddLoadFixtures(my_load_beginning, self)
            PlotPolygon(my_maze.bodies[len(my_maze.bodies) - 1], my_maze.arena_height,
                        cm.rainbow((np.linspace(0, 1, 5)))[4], 'fill')

        if not self.free:
            contact = self.play(1, 'contact')[1]
            breakpoint()
            # plt.plot(self.position_zoned[:,0], self.position_zoned[:,1], 'g.', markersize=3) #draw the zoned path
            fig.set_size_inches(my_maze.arena_length / 2, my_maze.arena_height / 2)
            plt.axis([0, my_maze.arena_length, 0, my_maze.arena_height])
            for i in range(len(contact) - 1):
                # if len(self.contact[i]) > 1 and self.contact[i].ndim == 1:
                if len(contact[i]) > 1:
                    con = np.vstack((con, np.array([contact[i][0], contact[i][1]])))
                    pos = np.vstack((pos, np.array([self.position[i][0], self.position[i][1]])))

            # Plot the slits
            breakpoint()
            for i in range(1, len(my_maze.bodies) - len(frames)):
                breakpoint()
                PlotPolygon(my_maze.bodies[i], my_maze.arena_height, 'k', 'fill')

            ''' adding strings... '''
            if kwargs.get('x_error') is not None:
                x_error = kwargs.get('x_error')
                y_error = kwargs.get('y_error')
                plt.text(1, my_maze.arena_height + 0.5, ("x_error = " + str("{:.2f}".format(x_error[0]))
                                                         + "cm   y_error = " + str(
                            "{:.2f}".format(y_error[0])) + "cm     " + self.filename
                                                         + "     " + str(int(self.timer())) + "s   "
                                                         + 'Frame:    ' + str(self.frames[0] + self.position.shape[0])),
                         fontsize=12
                         )
            else:
                plt.text(1, my_maze.arena_height + 0.5,
                         ("     " + self.filename),
                         fontsize=12,
                         )
            if kwargs.get('addstring') is not None:
                plt.text(my_maze.arena_length - 5, my_maze.arena_height + 0.5,
                         kwargs['addstring'],
                         fontsize=12,
                         )

            # plt.plot(np.append(mymaze.zone[:, 0],mymaze.zone[0,0]),
            #          np.append((mymaze.zone[:, 1]),(mymaze.zone[0, 1])),
            #           'y' + '-', markersize=3)
            ax.set_xticks(np.arange(0, my_maze.arena_length, 1))
            ax.set_yticks(np.arange(0, my_maze.arena_height, 1))
        else:
            ''' this is only for free motion '''
            plt.text(1, my_maze.arena_height + 0.5,
                     ("     " + self.filename) +
                     "     " + str(self.timer()) + "s",
                     fontsize=12,
                     )
            fig.set_size_inches(30, 15)
            plt.axis([0, 60, 0, 30])
        plt.show()  # show the plot....
        # if saver:
        #     from os import getcwd
        #     c = getcwd()
        #     fig.savefig(c + path.sep + 'Trajectory_Images' + path.sep + self.filename+'.pdf', format='pdf')

        # def Pdf2Png(pdf_name):
        #     pages = convert_from_path(pdf_name + '.pdf', 72)
        #     for page in pages:
        #         page.save(pdf_name + '.png', 'PNG')
        # Pdf2Png(self.filename)

    def Inspect(self, **kwargs):
        from Analysis_Functions.Velocity import velocity_x, max_Vel_angle, max_Vel_trans

        complaints = []
        if not (self.shape in ['RASH', 'LASH', 'H', 'I', 'T', 'SPT']):
            complaints = complaints + ['Your shape is not a real shape!']
            print(complaints[-1])
            breakpoint()

        # from bundle import sizes
        # if not (self.shape in ['H', 'I', 'T'] and self.size in sizes['ant'])\
        #     and not (self.shape in ['RASH', 'LASH', 'SPT'] and self.size in sizes['ant'].remove('XS'))\
        #     and not self.size in sizes['human']:
        #     print(str(self.size))
        #     complaints = complaints + ['This shape does not have a valid size']
        #     print(complaints[-1])
        #     breakpoint()

        if not (hasattr(self, 'free')):
            self.free = False
            complaints = complaints + ['added free']

        # print('VideoChain' + str(self.VideoChain))

        if self.filename.endswith('(part 1)') and len(self.VideoChain) < 2:
            print('You need to connect this!!')

        if not (self.solver == 'ant' or self.solver == 'human'):
            complaints = complaints + ['who is your solver ?']

        for i in range(len(self.VideoChain)):
            if not (self.old_filenames(i) in listdir(
                    MatlabFolder(self.solver, self.size, self.shape, self.free))) and not (
                    'CONNECTOR' in self.old_filenames(i)) and not ('_sim_' in self.old_filenames(i)):
                complaints = complaints + ['Why is ' + self.old_filenames(i) + ' not in original files? ']

        if not (hasattr(self, 'falseTracking')):
            self.falseTracking = [[]]
            complaints = complaints + ['added false tracking']
            if self.falseTracking == []:
                self.falseTracking = [[]]

            if len(self.falseTracking) != len(self.VideoChain) or type(self.falseTracking[0][0] == int):
                complaints = complaints + ['Your false tracking is not an adequate list!!']
                print(complaints[-1])
                breakpoint()

        if not self.free and not isinstance(self.x_error, list):
            complaints = complaints + ['Your error is not a list!!']
            print(complaints[-1])
            breakpoint()

        elif len(self.x_error) != len(self.VideoChain):
            complaints = complaints + ['Your error list does not have the right length']
            breakpoint()
            print(complaints[-1])

            # print('fps = ' + str(self.fps))
        if not (self.fps in [25, 50, 30]):
            self.fps = int(input('fps = '))
            complaints = complaints + ['fps where weird']
            breakpoint()
        # print('x_error = ' + str(self.x_error) + ', y_error = ' + str(self.y_error)+ ', angle_error = ' + str(
        # self.angle_error))

        if self.position.shape[0] != self.frames.shape[0]:
            breakpoint()
            if self.position.shape[0] - 1 == self.frames.shape[0]:
                self.frames = np.hstack((self.frames, [self.frames[-1] + 1]))
                complaints = complaints + ['Added 1 frame...']
            else:
                complaints = complaints + ['Check the dimensions of your position and your frames...']
                breakpoint()

            print(complaints[-1])

        if np.isnan(np.sum(self.position)):
            print('You have NaN values in your position')
            breakpoint()

        """Here, we correct wrong tracking... """
        if self.solver == 'ant':
            max_xy_vel = np.max(abs(velocity_x(self, 0.2, 'x', 'y')))
            if max_xy_vel > max_Vel_trans[self.size]:
                complaints = complaints + ['You have velocity ' + str(max_xy_vel) + 'cm/s at frame ' +
                                           str(self.frames[np.argmax(velocity_x(self, 0.2, 'x', 'y')[0])])]
                print(complaints[-1])

            if self.angle.shape != self.frames.shape:
                complaints = complaints + ['Check the dimensions of your angle and your frames...']
                print(complaints[-1])
                breakpoint()

            max_angle_vel = np.max(abs(velocity_x(self, 0.2, 'angle')))
            if max_angle_vel > max_Vel_angle[self.size]:
                complaints = complaints + ['You have angular velocity ' + str(max_angle_vel) + 'rad/s at frame ' +
                                           str(self.frames[np.argmax(velocity_x(self, 0.2, 'angle')[0])]) +
                                           ' or ConnectAngle(self.angle[:,0])']
                where = np.argmax(abs(velocity_x(self, 0.2, 'angle')))
                plt.plot(np.arange(where - 100, where + 100), self.angle[where - 100: where + 100])
                plt.show()
                print(complaints[-1])
                breakpoint()

        if not self.winner and not (self.shape == 'SPT'):
            complaints = complaints + ['I lost, even though I was not a Special T?']
            print(complaints[-1])

        if not (hasattr(self, 'free')):
            complaints = complaints + ['We added attribute free']
            self.free = False

        if self.solver == 'ants' and not (hasattr(self, 'tracked_frames')) and len(self.tracked_frames) != 2:
            if len(self.tracked_frames) != len(self.VideoChain):
                complaints = complaints + ['Your tracked_frames has the wrong dimensions!']
                breakpoint()

            for trac in self.tracked_frames:
                if len(trac) != 2 or trac[0] > trac[1]:
                    breakpoint()
                    complaints = complaints + ['Which frames where tracked?']

        if 'saver' in kwargs and kwargs['saver']:
            Save(self)
        return self, complaints

    def step(self, my_load, i):
        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]
        return

    def play(self, *args, interval=1, **kwargs):
        from copy import deepcopy
        x = deepcopy(self)

        if hasattr(x, 'contact'):
            delattr(x, 'contact')

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if 'indices' in kwargs.keys():
            f1, f2 = int(kwargs['indices'][0]), int(kwargs['indices'][1]) + 1
            x.position, x.angle = x.position[f1:f2, :], x.angle[f1:f2]
            x.frames = x.frames[int(f1):int(f2)]

        if 'attempt' in args:
            if not ('moreBodies' in kwargs.keys()):
                from Setup.Attempts import AddAttemptZone
                kwargs['moreBodies'] = AddAttemptZone

        if 'L_I_425' in x.filename:
            args = args + ('L_I1',)

        return Box2D_GameLoops.MainGameLoop(x, *args, display=True, interval=interval, **kwargs)
