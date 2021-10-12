from Directories import MatlabFolder
import scipy.io as sio
import numpy as np
from os import path
from trajectory_inheritance.forces import Forces, sheet


participant_number = {'Small Near': 1, 'Small Far': 1, 'Medium': 9, 'Large': 26}

angle_shift = {  # the number keys describe the names, zero based (Avirams numbering is 1 based)
    'Medium': {0: 0,
               1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
               4: np.pi, 5: np.pi,
               6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2},

    # the number keys describe the names, based on 0=A, ..., 25=Z.
    'Large': {3: np.pi / 2, 4: np.pi / 2, 5: np.pi / 2, 6: np.pi / 2, 7: np.pi / 2, 8: np.pi / 2, 9: np.pi / 2,
              10: 0,
              11: np.pi / 2,
              12: np.pi, 13: np.pi, 14: np.pi, 15: np.pi,
              16: -np.pi / 2,
              17: np.pi / 2,  # this seems to be a mistake in Avirams code
              18: -np.pi / 2, 19: -np.pi / 2, 20: -np.pi / 2, 21: -np.pi / 2, 22: -np.pi / 2, 23: -np.pi / 2,
              24: -np.pi / 2,
              25: -np.pi / 2,
              0: 0, 1: 0,
              2: np.pi / 2,
              },
}


class Humans_Frame:
    def __init__(self, size):
        self.position = np.zeros((participant_number[size], 2))
        self.angle = np.zeros(participant_number[size])
        self.carrying = np.zeros((participant_number[size]))
        self.major_axis_length = list()
        # self.forces = list()


class Humans:
    def __init__(self, x):
        self.filename = x.filename
        self.excel_index = self.get_excel_worksheet_index()
        self.VideoChain = [x.filename]
        self.frames = list()
        self.size = x.size
        self.number = len(self.gender())

        # contains list of occupied sites, where site A carries index 0 and Z carries index 25.
        self.occupied = list(self.gender().keys())

        self.matlab_human_loading(x)
        self.angles = self.get_angles()
        self.forces = Forces(self, x)
        self.gender_string = self.gender()

    def get_excel_worksheet_index(self):
        number_exp = [i for i in range(1, int(sheet.dimensions.split(':')[1][1:]))
                      if sheet.cell(row=i, column=1).value is not None][-1]

        name_list = self.filename.split('_')
        indices = []
        for i in range(2, number_exp + 1):
            if len([ii for ii in range(len(name_list))
                    if i <= number_exp and sheet.cell(row=i, column=1).value is not None and
                       name_list[ii] in sheet.cell(row=i, column=1).value.split('_')]) > 1:
                indices.append(i)
        if len(indices) == 1:
            return indices[0]
        elif len(name_list[-1]) > 1:  # has to be the first run
            return indices[int(np.argmin([sheet.cell(row=index, column=6).value for index in indices]))]
        elif len(name_list[-1]) == 1:
            return indices[np.argsort([sheet.cell(row=index, column=6).value
                                       for index in indices])[int(name_list[-1]) - 1]]
        elif len(indices) == 0:
            print('cant find your movie')

    def matlab_human_loading(self, x):
        file = sio.loadmat(MatlabFolder(x.solver, x.size, x.shape, False) + path.sep + self.VideoChain[0])
        matlab_cell = file['hats']

        Medium_id_correction_dict = {1: 1, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2}

        for i, Frame in enumerate(matlab_cell):
            data = Frame[0]

            # to sort the data
            humans_frame = Humans_Frame(self.size)

            # if identities are given
            if data.shape[1] > 4:
                data = data[data[:, 4].argsort()]

            if x.size in ['Medium', 'Large']:

                if x.filename == 'large_20210419100024_20210419100547':
                    for false_reckog in [8., 9.]:
                        index = np.where(data[:, 4] == false_reckog)
                        data = np.delete(data, index, 0)

                # correct the wrong hat identities
                if x.size == 'Medium':
                    data = data[np.vectorize(Medium_id_correction_dict.get)(data[:, 4]).argsort()]
                    data[:, 4] = np.vectorize(Medium_id_correction_dict.get)(data[:, 4])
                if x.size == 'Large':
                    pass

                data = data[data[:, 4].argsort()]

                # tracked participants have a carrying boolean, and an angle to their force meter
                humans_frame.position[self.occupied] = data[:, 2:4] + [x.x_error[0], x.y_error[0]]
                humans_frame.carrying[self.occupied] = data[:, 5]

                # angle to force meter
                humans_frame.angle[self.occupied] = data[:, 6] * np.pi / 180 + x.angle_error[0]

            self.frames.append(humans_frame)
        return

    def get_angles(self):
        return np.array([fr.angle for fr in self.frames])

    def averageCarrierNumber(self):
        return self.number

    def correlation(self, players=None, frames=None):
        """
        :param frames: forces in what frames are you interested in finding their correlation
        :param players: list of players that you want to find correlation for
        :return: nxn correlation matrix, where n is either the length of the kwarg players or the number of forcemeters
        on the shape
        """
        if frames is None:
            frames = [0, len(self.frames)]
        if players is None:
            players = self.occupied

        forces_in_x_direction = [
            self.forces.array[:, player][slice(*frames, 1)] * np.cos(angle_shift[self.size][player])
            for player in players]
        correlation_matrix = np.corrcoef(np.stack(forces_in_x_direction))
        return correlation_matrix

    def gender(self):
        """
        return dict which gives the gender of every participant. The keys of the dictionary are indices of participants,
        where participant A has index 0, B has index 1, ... and Z has index 25. This is different from the counting in
        Aviram's movies!!
        """
        gender_string = list(sheet.cell(row=self.excel_index, column=17).value)

        if len(gender_string) != participant_number[self.size] \
                and not (len(gender_string) in [1, 2] and self.size == 'Medium'):
            print('you have an incorrect gender string in ' + str(self.excel_index))
        return {i: letter for i, letter in enumerate(gender_string) if letter != '0'}

