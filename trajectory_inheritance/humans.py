from abc import ABC
from Box2D import b2Vec2
from Directories import MatlabFolder
import scipy.io as sio
import numpy as np
from os import path
from trajectory_inheritance.forces import Forces, sheet
from trajectory_inheritance.participants import Participants
from Setup.Maze import Maze
from PhysicsEngine.drawables import colors, Circle, Line

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


def get_excel_worksheet_index(filename) -> int:
    """
    :param filename: filename of the tracked movie (like 'medium_20211006172352_20211006172500')
    :return: index of the excel worksheet line
    """
    number_exp = [i for i in range(1, int(sheet.dimensions.split(':')[1][1:]))
                  if sheet.cell(row=i, column=1).value is not None][-1]

    times_list = filename.split('_')[1:3]
    indices = []

    for i in range(2, number_exp + 1):
        in_filled_lines = (i <= number_exp and sheet.cell(row=i, column=1).value is not None)
        old_filename_times = sheet.cell(row=i, column=1).value.split(' ')[0].split('_')
        if len([ii for ii in range(len(times_list))
                if in_filled_lines and times_list[ii] in old_filename_times]) > 1:
            indices.append(i)

    if len(indices) == 1:
        return indices[0]
    elif len(times_list[-1]) > 1:  # has to be the first run
        return indices[int(np.argmin([sheet.cell(row=index, column=6).value for index in indices]))]
    elif len(times_list[-1]) == 1:
        return indices[np.argsort([sheet.cell(row=index, column=6).value
                                   for index in indices])[int(times_list[-1]) - 1]]
    elif len(indices) == 0:
        print('cant find your movie')


class Humans_Frame:
    def __init__(self, size):
        self.position = np.zeros((participant_number[size], 2))
        self.angle = np.zeros(participant_number[size])
        self.carrying = np.zeros((participant_number[size]))
        self.major_axis_length = list()
        # self.forces = list()


class Humans(Participants, ABC):
    def __init__(self, x, color=''):
        super().__init__(x, color='')

        self.excel_index = get_excel_worksheet_index(self.filename)
        self.number = len(self.gender())

        # contains list of occupied sites, where site A carries index 0 and Z carries index 25 (for size 'large').
        self.occupied = list(self.gender().keys())

        self.matlab_loading(x)
        self.angles = self.get_angles()
        self.positions = self.get_positions()
        self.gender_string = self.gender()
        if sheet.cell(row=self.excel_index, column=19).value != '/':
            self.forces = Forces(self, x)

    def matlab_loading(self, x) -> None:
        file = sio.loadmat(MatlabFolder(x.solver, x.size, x.shape) + path.sep + self.VideoChain[0])
        matlab_cell = file['hats']

        Medium_id_correction_dict = {1: 1, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2}

        # # for falsely tracked angles
        # if x.filename in ['medium_20201223130749_20201223131147', 'medium_20201223125622_20201223130532']:
        #     my_maze = Maze(x)

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
                if data[:, 2:4].shape[0] != len(self.occupied):
                    data = self.add_missed_hat(matlab_cell[i-1][0], data)
                    matlab_cell[i][0] = data

                humans_frame.position[self.occupied] = data[:, 2:4] + np.array([x.x_error[0], x.y_error[0]])

                # if force meters were installed, then only carrying boolean and angle were included in .mat file
                if data.shape[1] > 5:
                    humans_frame.carrying[self.occupied] = data[:, 5]

                    # angle to force meter
                    if x.filename not in ['medium_20201223130749_20201223131147',
                                          'medium_20201223125622_20201223130532']:
                        # here, the identities were given wrong in the tracking
                        humans_frame.angle[self.occupied] = data[:, 6] * np.pi / 180 + x.angle_error[0]
                    else:
                        humans_frame.angle[self.occupied] = data[:, 6] * np.pi / 180 + x.angle_error[0]
                        my_maze = Maze(x)
                        my_maze.set_configuration(x.position[i], x.angle[i])
                        humans_frame.angle[self.occupied] = self.angle_to_forcemeter(humans_frame.position, my_maze,
                                                                                     x.angle[i], x.size)

            self.frames.append(humans_frame)
        return

    def add_missed_hat(self, data_full, data_missing):
        return data_full

    def angle_to_forcemeter(self, positions, my_maze, angle, size) -> np.ndarray:
        r = positions[self.occupied] - self.force_attachment_positions(my_maze)[self.occupied]
        angles_to_normal = np.arctan2(r[:, -1], r[:, 0]) - \
                           np.array([angle_shift[size][occ] for occ in self.occupied]) - angle
        return angles_to_normal

    def get_angles(self) -> np.ndarray:
        return np.array([fr.angle for fr in self.frames])

    def get_positions(self) -> np.ndarray:
        return np.array([fr.position for fr in self.frames])

    def averageCarrierNumber(self) -> int:
        return self.number

    def correlation(self, players=None, frames=None) -> np.ndarray:
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
            self.forces.abs_values[:, player][slice(*frames, 1)] * np.cos(angle_shift[self.size][player])
            for player in players]
        correlation_matrix = np.corrcoef(np.stack(forces_in_x_direction))
        return correlation_matrix

    def gender(self) -> dict:
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

    def draw(self, display) -> None:
        for part in self.occupied:
            Circle(self.positions[display.i, part], 0.1, colors['hats'], hollow=False).draw(display)
            if hasattr(self, 'forces'):
                force_attachment = self.force_attachment_positions(display.my_maze)
                Circle(force_attachment[part], 0.05, (0, 0, 0), hollow=False).draw(display)
                Line(self.positions[display.i, part], force_attachment[part], (0, 0, 0)).draw(display)

    def force_attachment_positions(self, my_maze):
        from Setup.Maze import centerOfMass_shift
        if self.size == 'Medium':
            # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
            [shape_height, shape_width, shape_thickness] = my_maze.getLoadDim()
            x29, x38, x47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4

            # (0, 0) is the middle of the shape
            positions = [[shape_width / 2, 0],
                         [x29, shape_thickness / 2],
                         [x38, shape_thickness / 2],
                         [x47, shape_thickness / 2],
                         [-shape_width / 2, shape_height / 4],
                         [-shape_width / 2, -shape_height / 4],
                         [x47, -shape_thickness / 2],
                         [x38, -shape_thickness / 2],
                         [x29, -shape_thickness / 2]]
            h = centerOfMass_shift * shape_width

        elif self.size == 'Large':
            [shape_height, shape_width, shape_thickness, short_edge] = my_maze.getLoadDim(short_edge=True)

            xMNOP = -shape_width / 2
            xLQ = xMNOP + shape_thickness / 2
            xAB = (-1) * xMNOP
            xCZ = (-1) * xLQ
            xKR = xMNOP + shape_thickness
            xJS, xIT, xHU, xGV, xFW, xEX, xDY = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

            yA_B = short_edge / 6
            yC_Z = short_edge / 2
            yDEFGHIJ_STUVWXY = shape_thickness / 2
            yK_R = shape_height / 10 * 2
            yL_Q = shape_height / 2
            yM_P = shape_height / 10 * 3
            yN_O = shape_height / 10

            # indices in comment describe the index shown in Aviram's tracking movie
            positions = [[xAB, -yA_B],  # 1, A
                         [xAB, yA_B],  # 2, B
                         [xCZ, yC_Z],  # 3, C
                         [xDY, yDEFGHIJ_STUVWXY],  # 4, D
                         [xEX, yDEFGHIJ_STUVWXY],  # 5, E
                         [xFW, yDEFGHIJ_STUVWXY],  # 6, F
                         [xGV, yDEFGHIJ_STUVWXY],  # 7, G
                         [xHU, yDEFGHIJ_STUVWXY],  # 8, H
                         [xIT, yDEFGHIJ_STUVWXY],  # 9, I
                         [xJS, yDEFGHIJ_STUVWXY],  # 10, J
                         [xKR, yK_R],  # 11, K
                         [xLQ, yL_Q],  # 12, L
                         [xMNOP, yM_P],  # 13, M
                         [xMNOP, yN_O],  # 14, N
                         [xMNOP, -yN_O],  # 15, O
                         [xMNOP, -yM_P],  # 16, P
                         [xLQ, -yL_Q],  # 17, Q
                         [xKR, -yK_R],  # 18, R
                         [xJS, -yDEFGHIJ_STUVWXY],  # 19, S
                         [xIT, -yDEFGHIJ_STUVWXY],  # 20, T
                         [xHU, -yDEFGHIJ_STUVWXY],  # 21, U
                         [xGV, -yDEFGHIJ_STUVWXY],  # 22, V
                         [xFW, -yDEFGHIJ_STUVWXY],  # 23, W
                         [xEX, -yDEFGHIJ_STUVWXY],  # 24, X
                         [xDY, -yDEFGHIJ_STUVWXY],  # 25, Y
                         [xCZ, -yC_Z],  # 26, Z
                         ]
            h = centerOfMass_shift * shape_width
        # centerOfMass_shift the shape...
        positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame
        return np.array(
            [np.array(my_maze.bodies[-1].GetWorldPoint(b2Vec2(r))) for r in positions])  # r vectors in the lab frame
