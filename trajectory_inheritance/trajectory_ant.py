from trajectory_inheritance.trajectory import Trajectory
from Directories import NewFileName
from copy import deepcopy
import numpy as np
import scipy.io as sio
from os import path
from Setup.Maze import Maze
from PhysicsEngine.Display import Display

length_unit = 'cm'

trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)


class Trajectory_ant(Trajectory):
    def __init__(self, size=None, shape=None, old_filename=None, free=False, fps=50, winner=bool, x_error=0, y_error=0,
                 angle_error=0, falseTracking=[]):

        filename = NewFileName(old_filename, size, shape, 'exp')

        super().__init__(size=size, shape=shape, solver='ant', filename=filename, fps=fps, winner=winner)
        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.free = free
        self.state = np.empty((1, 1), int)
        self.different_dimensions = 'L_I_425' in self.filename

    # def __del__(self):
    #     remove(ant_address(self.filename))

    def new2021(self):
        """
        I restarted experiments and altered the maze dimensions for the S, M, L and XL SPT.
        I am keeping track of the movies, that have these altered maze dimensions.
        :return: bool.
        """
        new_starting_conditions = [str(x) for x in range(46300, 48100, 100)]
        return np.any([new_starting_condition in self.filename
                       for new_starting_condition in new_starting_conditions])

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

    def old_filenames(self, i):
        old = None
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
        return old

    def matlabFolder(self):
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not self.free:
            return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
                self.shape] + path.sep + self.size + path.sep + 'Output Data'
        if self.free:
            return trackedAntMovieDirectory + path.sep + 'Free' + path.sep + 'Output Data' + path.sep + \
                   shape_folder_naming[self.shape]

    def matlab_loading(self, old_filename):
        if not (old_filename == 'XLSPT_4280007_XLSpecialT_1_ants (part 3).mat'):
            file = sio.loadmat(self.matlabFolder() + path.sep + old_filename)

            # if 'Direction' not in file.keys():
            #     file['Direction'] = 'R2L'
            #     print('Direction = R2L')
                # file['Direction'] = None

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
                    # error in the end_screen... )

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
                np.matrix.transpose(file['shape_orientation'][:] * np.pi / 180 + self.angle_error)[0]
            #
            # if file['Direction'] == 'R2L':
            #     shape_orientation = (shape_orientation + np.pi) % np.pi

        else:
            import h5py
            with h5py.File(self.matlabFolder() + path.sep + old_filename, 'r') as f:
                load_center = np.matrix.transpose(f['load_center'][:, :])
                load_center[:, 0] = load_center[:, 0] + self.x_error
                load_center[:, 1] = load_center[:, 1] + self.y_error
                self.frames = np.matrix.transpose(f['frames'][:])[0]
                # # Angle accounts for shifts in the angle of the shape.... (manually, by watching the movies)
                shape_orientation = (f['shape_orientation'][:] * np.pi / 180 + self.angle_error[0])[0]

                # # if not('Direction' in file.keys()) and not(self.shape == 'T' and self.size == 'S'):

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)

        from Analysis.Velocity import check_for_false_tracking
        self.falseTracking = [check_for_false_tracking(self)]
        self.falseTracker()
        self.interpolate_over_NaN()

        return

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

    def load_participants(self):
        from trajectory_inheritance.ants import Ants
        self.participants = Ants(self)

    def averageCarrierNumber(self):
        self.load_participants()
        self.participants.averageCarrierNumber()

    def play(self, indices=None, wait=0, ps=None, step=1, videowriter=False):
        """
        Displays a given trajectory_inheritance (self)
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

        my_maze = Maze(x, new2021=self.new2021())
        return x.run_trj(my_maze, display=Display(x, my_maze, wait=wait, ps=ps, videowriter=videowriter))
