from trajectory_inheritance.trajectory import Trajectory
import numpy as np
from os import path, listdir
import scipy.io as sio
from trajectory_inheritance.humans import Humans
from Directories import MatlabFolder

trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
length_unit = 'm'


class Trajectory_human(Trajectory):
    # TODO: Check the fps of the human pickles (that they are all 30 fr/s)!
    def __init__(self, size=None, shape=None, filename=None, fps=30, winner=bool, x_error=None,
                 y_error=None, angle_error=None, falseTracking=None, VideoChain=str(), forcemeter=bool()):

        super().__init__(size=size, shape=shape, solver='human', filename=filename, fps=fps, winner=winner)

        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.state = np.empty((1, 1), int)
        self.VideoChain = VideoChain
        self.communication = self.communication()
        self.forcemeter = forcemeter

    def matlab_loading(self, old_filename):
        folder = MatlabFolder(self.solver, self.size, self.shape)

        if old_filename + '.mat' in listdir(folder):
            file = sio.loadmat(folder + path.sep + old_filename + '.mat')
        else:
            raise Exception('Cannot find ' + old_filename + '.mat' + ' in ' + str(folder))

        load_center = file['load_CoM'][:, 2:4]
        load_center[:, 0] = load_center[:, 0] + self.x_error
        load_center[:, 1] = load_center[:, 1] + self.y_error
        shape_orientation = np.matrix.transpose(file['orientation'][:] * np.pi / 180 + self.angle_error)[0]

        self.frames = np.linspace(1, load_center.shape[0], load_center.shape[0]).astype(int)

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)
        self.interpolate_over_NaN()

    def communication(self):
        from trajectory_inheritance.humans import get_excel_worksheet_index
        from trajectory_inheritance.forces import get_sheet
        index = get_excel_worksheet_index(self.filename)
        return get_sheet().cell(row=index, column=5).value == 'C'

    def load_participants(self):
        if not hasattr(self, 'participants'):
            self.participants = Humans(self)

    def averageCarrierNumber(self):
        self.participants.averageCarrierNumber()
