from Directories import MatlabFolder, home
import scipy.io as sio
import numpy as np
from copy import deepcopy
from os import path
from trajectory_inheritance.participants import Participants
from trajectory_inheritance.get import get
from scipy.signal import medfilt
from Setup.Maze import Maze
from tqdm import tqdm


class Ants_Frame:
    def __init__(self, position, angle, carrying):
        self.position = position
        self.angle = angle
        self.carrying = carrying  # 0 if not connected, 1 if connected on outer rim of shape, 2 if connected on inner
        # rim of shape

    def carrierCount(self):
        return len([pos for [pos, car] in zip(self.position, self.carrying) if car])


class Ants(Participants):
    def __init__(self, x):
        super().__init__(x)
        self.pix2cm = np.NaN
        self.matlab_loading(x)
        self.angles = self.get_angles()
        self.positions = self.get_positions()

    def __add__(self, other):
        if self.filename != other.filename:
            print('ants don''t belong together!!')
        ants_combined = deepcopy(self)
        ants_combined.frames = ants_combined.frames + other.frames
        ants_combined.VideoChain = ants_combined.VideoChain + other.VideoChain
        return ants_combined

    def is_ant(self, major_axis_length) -> bool:
        if 0.12 < major_axis_length * self.pix2cm < 0.35:
            return True
        else:
            return False

    def carriers_in_frame(self, fps):
        # many short-lived attachment and detachment events. We get rid of them with a median smooth.
        medfilt_cc = medfilt([len(frame.position) for frame in self.frames], 2 * fps + 1)
        return medfilt_cc

    def carriers_attached(self, fps):
        # many short-lived attachment and detachment events. We get rid of them with a median smooth.
        medfilt_cc = medfilt([np.sum(frame.carrying) for frame in self.frames], 2 * fps + 1)
        return medfilt_cc

    def is_attached(self, p_ant, p_shape, max_distance):
        return np.linalg.norm(p_ant - p_shape) < max_distance

    def averageCarrierNumber(self):
        if len(self.frames):
            return np.ceil(np.mean(self.carriers_attached(0)))
        else:
            return None

    def number_of_attached(self):
        pass

    # def k_on1(self, fps):
    #     """
    #     Attachment events per second
    #     """
    #     cc = self.carriers_in_frame(fps)
    #     total_time = len(cc)/fps
    #     attachment_events = 0
    #     for fr1, fr2 in zip(cc[:-1], cc[1:]):
    #         attachment_events += max(0, fr2 - fr1)
    #     return attachment_events / (self.averageCarrierNumber() * total_time)

    # def k_on(self, fps):
    #     """
    #     Attachment events per second # TODO: these are NOT attachments, but instead counts how many ants are in the frame!!!
    #     """
    #     cc = self.carriers_in_frame(fps)
    #     attachment_events = [0]
    #     for i, (fr1, fr2) in enumerate(zip(cc[:-1], cc[1:])):
    #         if fr2 > fr1:
    #             dt = i/fps - np.sum(attachment_events)
    #             attachment_events += [dt/(fr2 - fr1) for _ in range(int(fr2 - fr1))]
    #     return (1/np.mean(attachment_events)) / self.averageCarrierNumber()
    #
    # def k_off(self, fps):
    #     """
    #     Detachment events per second
    #     """
    #     cc = self.carriers_in_frame(fps)
    #     detachment_events = [0]
    #     for i, (fr1, fr2) in enumerate(zip(cc[:-1], cc[1:])):
    #         if fr2 < fr1:
    #             dt = i/fps - np.sum(detachment_events)
    #             detachment_events += [dt/(fr1 - fr2) for _ in range(int(fr1 - fr2))]
    #     return (1/np.mean(detachment_events)) / self.averageCarrierNumber()

    def matlab_loading(self, x):
        print(x.filename)
        matlab_cell = np.zeros(shape=(0, 1))
        ant_frames = []

        first_position_error = np.zeros(shape=(0, 2))

        for i in range(len(x.VideoChain)):
            filename = x.old_filenames(i)
            if 'r).mat' in filename and ant_frames[-1] > 50000:
                pass  # this means, that we retracked the load. And then we retracked the ants, adn the ants do not have
                # a separate 1r file. So we skip this one.

            else:
                if 'CONNECTOR' in filename:
                    # self.frames = self.frames + [self.frames[-1] for _ in range(x.number_of_frames_in_part(i))]
                    raise ValueError('Could not find connector file ' + filename)

                elif path.isfile(path.join(home, 'Analysis', 'AttachmentStatistics', 'retrackedMatlabFiles', filename)):
                    file = sio.loadmat(path.join(home, 'Analysis', 'AttachmentStatistics', 'retrackedMatlabFiles', filename))

                elif path.isfile(MatlabFolder('ant', x.size, x.shape) + path.sep + filename):
                    file = sio.loadmat(MatlabFolder('ant', x.size, x.shape) + path.sep + filename)

                else:
                    raise ValueError('Could not find file ' + filename)

                if 'Direction' not in file.keys() and x.shape.endswith('ASH'):
                    # file['Direction'] = input('L2R or R2L  ')
                    file['Direction'] = None

                if x.shape.endswith('ASH'):
                    if 'R2L' == file['Direction']:
                        if x.shape == 'LASH':
                            x.shape = 'RASH'
                            x.filename.replace('LASH', 'RASH')
                            x.VideoChain = [name.replace('LASH', 'RASH') for name in self.VideoChain]

                        else:
                            x.shape = 'LASH'
                            x.filename.replace('RASH', 'LASH')
                            x.VideoChain = [name.replace('RASH', 'LASH') for name in self.VideoChain]

                        if x.angle_error[0] == 0:
                            if x.shape == 'LASH':
                                x.angle_error = 2 * np.pi * 0.11 + x.angle_error
                            if x.shape == 'RASH':
                                x.angle_error = -2 * np.pi * 0.11 + x.angle_error
                                # # For all the Large Asymmetric Hs I had 0.1!!! (I think, this is why I needed the
                                # error in the end_screen... )

                            if x.shape == 'LASH' and self.size == 'XL':  # # It seems like the exit walls are a bit
                                # crooked, which messes up the contact tracking
                                x.angle_error = 2 * np.pi * 0.115 + x.angle_error
                            if x.shape == 'RASH' and self.size == 'XL':
                                x.angle_error = -2 * np.pi * 0.115 + x.angle_error
                if i == 0:
                    overall_fps = (file['frames'][0][1] - file['frames'][0][0])
                    scale = 1
                else:
                    if (file['frames'][0][-1] - file['frames'][0][-2]) != overall_fps:
                        scale = overall_fps/(file['frames'][0][-1] - file['frames'][0][-2])
                        if scale < 1:
                            raise ValueError('The FPS of the videos are not the same!')
                        scale = int(scale)
                    else:
                        scale = 1
                ant_frames += file['frames'][0].tolist()[::scale]
                new = file['ants'][::scale]
                error = file['load_center'][0] - x.position[0]
                matlab_cell = np.vstack([matlab_cell, new])
                first_position_error = np.vstack([first_position_error, [error for _ in range(len(new))]])

        # if len(matlab_cell) != len(frames):
        #     if len(matlab_cell) > frames[-1] - frames[0]:
        #         nth = np.round(len(matlab_cell) / frames[-1] - frames[0]).astype(int)
        #         matlab_cell = matlab_cell[::nth, :]
        #     elif len(matlab_cell) < len(frames):
        #         raise ValueError('Not enough frames in matlab file')
        #         nth = np.round(len(frames) / len(matlab_cell)).astype(int)
        #         matlab_cell = np.repeat(matlab_cell, nth, axis=0)

        # maze = Maze(x)
        self.pix2cm = file['pix2cm']
        self.frames = []

        matlab_cell = np.squeeze(matlab_cell)

        if x.filename == 'S_SPT_5190017_SSpecialT_1_ants':  # this file I later shortened, so I need to cut the matlab
            # cell
            matlab_cell = matlab_cell[:len(x.frames)]
            ant_frames = ant_frames[:len(x.frames)]

        if ant_frames[0] == x.frames[0] and abs(ant_frames[-1] - x.frames[-1])/x.fps < 50:
            pass
        if abs(len(matlab_cell) / len(x.frames) - 1) > 0.001:
            nth = ((x.frames[1] - x.frames[0])/(ant_frames[1] - ant_frames[0])).astype(int)
            matlab_cell = matlab_cell[::nth]
            if nth not in [5, 50, 10]:
                raise ValueError('Check something here')

        for data, pos_error in tqdm(zip(matlab_cell, first_position_error), desc='Loading ants in ' + filename):
            if data.size != 0:
                # real_ants = [self.is_ant(mal) for mal in data[:, 6]]
                if self.size is not 'S':
                    real_ants = np.ones(data.shape[0]).astype(bool)
                else:
                    # this is due to some black spot in the maze. It is always recognized as an ant.
                    real_ants = ~np.logical_and(np.logical_and(1.2 < data[:, 2], data[:, 2] < 1.5), data[:, 3] < 0.15)

                # position = data[real_ants, 2:4] + np.array([1.07, 0])
                position = data[real_ants, 2:4] - pos_error
                if type(x.angle_error) == list and len(x.angle_error) > 1:
                    x.angle_error = x.angle_error[0]
                angle = data[real_ants, 5] * np.pi / 180 + x.angle_error
                # is_attached = [self.is_attached(p_ant, p_shape, max_distance)
                #                for p_ant, p_shape in zip(position, x.position)]
                carrying = data[real_ants, 4]
                ants_frame = Ants_Frame(position, angle, carrying)
            else:
                ants_frame = Ants_Frame(np.array([]), np.array([]), [])
            self.frames.append(ants_frame)

        if x.old_filenames(0) == 'XLSPT_4280007_XLSpecialT_1_ants (part 3).mat':
            raise ValueError('This is a special case, the file was just to big')
            # import h5py
            # with h5py.File(
            #         MatlabFolder(x.solver, x.size, x.shape) + path.sep + x.old_filename,
            #         'r') as f:
            #     load_center = np.matrix.transpose(f['load_center'][:, :])

    def get_angles(self) -> list:
        return [fr.angle if fr is not None else None for fr in self.frames]

    def get_positions(self) -> list:
        return [fr.position if fr is not None else None for fr in self.frames]

