from trajectory_inheritance.trajectory import Trajectory
import numpy as np
from os import path
import scipy.io as sio
from trajectory_inheritance.humans import Humans

trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
length_unit = 'm'


class Trajectory_human(Trajectory):
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, x_error=None,
                 y_error=None, angle_error=None, falseTracking=None, VideoChain=str(), forcemeter=bool()):

        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.state = np.empty((1, 1), int)
        self.VideoChain = VideoChain
        self.communication = self.communication()
        self.forcemeter = forcemeter

    def matlabFolder(self):
        return trackedHumanMovieDirectory + path.sep + self.size + path.sep + 'Data'

    def matlab_loading(self, old_filename):
        file = sio.loadmat(self.matlabFolder() + path.sep + old_filename)
        load_center = file['load_CoM'][:, 2:4]
        load_center[:, 0] = load_center[:, 0] + self.x_error
        load_center[:, 1] = load_center[:, 1] + self.y_error
        shape_orientation = np.matrix.transpose(file['orientation'][:] * np.pi / 180 + self.angle_error[0])[0]

        self.frames = np.linspace(1, load_center.shape[0], load_center.shape[0]).astype(int)

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)
        self.interpolate_over_NaN()

    def participants(self):
        return Humans(self)

    def step(self, my_load, i, **kwargs):
        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]

    def averageCarrierNumber(self):
        self.participants().averageCarrierNumber()

