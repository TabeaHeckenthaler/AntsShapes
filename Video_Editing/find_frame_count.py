import cv2
import os
import scipy.io as sio
import pandas as pd
import numpy as np


folder_exc_mat = {'large': 'Large maze', 'medium': 'Medium maze', 'small': 'Small near maze', 'small2': 'Small far maze'}
sizes_exc_mat = {'L': 'large', 'M': 'medium', 'SN': 'small', 'SF': 'small2'}
force_meters = {'large': True, 'medium': True, 'small': False, 'small2': False}
switching = {'large': False, 'medium': False, 'small': False, 'small2': False}


class Experiment:
    def __init__(self, pd_Series):
        self.index = None
        self.size = sizes_exc_mat[pd_Series['Maze Size']]
        self.movie_name = pd_Series['Video File Name'].split('\n')[0].strip()
        self.init_frame = pd_Series['Initial Frame']
        self.end_frame = pd_Series['End Frame']
        self.movie_extensions = pd_Series['Video File Name'].split('\n')[1:]
        self.num_part = int(pd_Series['Group Size'])

    def matlab_line(self):
        line = np.array([
            np.array(self.movie_name),
            np.array(self.frames_first_movie()),
            np.array(self.size),
            np.array(self.video_extensions()),
            np.array(force_meters[self.size]),
            np.array(switching[self.size]),
            np.array(switching[self.num_part])
                  ], dtype=object)

        # array([array(['NVR_ch1_main_20220527112227_20220527112931'], dtype='<U42'),
        #        array([[900, 7230]], dtype=uint16), array(['large'], dtype='<U5'),
        #        array([], shape=(1, 0), dtype=float64), array([[1]], dtype=uint8),
        #        array([[0]], dtype=uint8)], dtype=object)

        return line

    def frames_first_movie(self) -> list:
        f1 = int(str(self.init_frame).split(',')[0])
        if len(self.movie_extensions) > 0:
            f2 = self.last_frame(self.movie_name)
        else:
            f2 = int(str(self.end_frame).split(',')[-1])
        return [f1, f2]

    def video_extensions(self) -> np.array:
        if len(self.movie_extensions) > 0:
            line = np.ndarray(shape=(0, 2))
            for name in self.movie_extensions:
                new_line = np.array([[
                    np.array(name + '.asf'),
                    np.array([1, self.last_frame(name)])
                ]], dtype=object)
                line = np.concatenate([line, new_line])
        else:
            line = []
        return line

    def last_frame(self, movie) -> int:
        video_directory = os.path.join('P:\\', 'Tabea', 'Human Experiments',
                                       'Raw Data and Videos', date, 'Videos', folder_exc_mat[self.size])
        address = os.path.join(video_directory, movie + '.asf')
        cap = cv2.VideoCapture(address)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if frames == -1:
            DEBUG = 1
        cap.release()
        return frames


not_trackable = ['NVR_ch1_main_20211107170147_20211107171803',
                 'NVR_ch4_main_20220606184501_20220606184827']


class ExperimentAdder:
    def __init__(self, excel_sheet, matlab_cell):
        self.excel_sheet = excel_sheet
        self.matlab_cell = matlab_cell

    def exp_missing_in_matlab_cell(self) -> list:
        movies_in_matlab = [f[0] for f in self.matlab_cell[:, 0]]
        movies_in_excel = self.movies_in_excel()
        return [m for m in movies_in_excel if m not in movies_in_matlab]
        # return set(movies_in_excel) - set(movies_in_matlab)

    def movies_in_excel(self):
        l = [f.split('\n')[0].split(' (')[0].strip() for f in list(self.excel_sheet['Video File Name'])]
        return [l1 for l1 in l if l1 not in not_trackable]


testable_dir = '\\\\phys-guru-cs\\ants\\Tabea\\Human Experiments\\Testable.xlsx'
matlab_dir = '\\\\phys-guru-cs\\ants\\Aviram\\Shapes Tracking Software\\Homo sapiens sapiens\\' \
             'Input Files\\human_shapes_video_data.mat'
date = '2022-06-06'
sizes = [size + ' maze' for size in ['Large', 'Medium', 'Small far', 'Small near']]

if __name__ == '__main__':
    video_data_cell = sio.loadmat(matlab_dir)['human_shapes_video_data_cell']
    excel = pd.read_excel(io=testable_dir, sheet_name='Human maze')

    exp_adder = ExperimentAdder(excel, video_data_cell)
    new_matlab_cell = exp_adder.matlab_cell.copy()
    to_add = exp_adder.exp_missing_in_matlab_cell()

    for movie_name in to_add:
        line = exp_adder.excel_sheet[exp_adder.excel_sheet['Video File Name'].
                                         apply(lambda x: x.split('\n')[0].split(' (')[0].strip()) == movie_name]
        print(line)
        assert line.shape[0] == 1, line
        exp = Experiment(line.iloc[0])
        m_line = exp.matlab_line()
        new_matlab_cell = np.vstack([new_matlab_cell, m_line])

    sio.savemat(matlab_dir.split('.')[0] + '_python' + '.mat', {'human_shapes_video_data_cell': new_matlab_cell})

    DEBUG = 1
