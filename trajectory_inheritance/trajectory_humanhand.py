from trajectory_inheritance.trajectory import Trajectory, length_unit
from os import path
import pickle
import numpy as np

trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'
length_unit = 'cm'


class Trajectory_humanhand(Trajectory):
    def __init__(self, size=None, shape=None, filename=None, fps=50, winner=bool, x_error=None, y_error=None,
                 angle_error=None, falseTracking=None):

        super().__init__(size=size, shape=shape, solver='humanhand', filename=filename, fps=fps, winner=winner)
        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.state = np.empty((1, 1), int)

    def matlabFolder(self):
        return trackedHumanHandMovieDirectory

    def matlab_loading(self, old_filename):
        load_center, angle, frames, videoChain = pickle.load(open(self.matlabFolder() + path.sep + self.filename + '.pkl', 'rb'))

        if len(load_center) == 2:
            self.position = np.array([load_center])
            self.angle = np.array([angle])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(angle)
        self.frames = np.array(frames)
        self.VideoChain = videoChain
        # self.interpolate_over_NaN()

    def load_participants(self):
        self.participants = Humanhand(self)

    def averageCarrierNumber(self):
        return 1

    def geometry(self) -> tuple:
        return 'MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx'

    def add_missing_frames(self, chain):
        pass


class Humanhand:
    def __init__(self, filename):
        self.filename = filename
        return
