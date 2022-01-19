from PS_Search_Algorithms.D_star_lite import run_dstar
from trajectory_inheritance.trajectory_ant import Trajectory_ant
from trajectory_inheritance.trajectory import get
import numpy as np
from scipy.ndimage import gaussian_filter
from Setup.Load import periodicity
from Analysis.Velocity import velocity


# def Connector_straight_line(file1, file2, con_frames: int = None):
#     # Instantiate a new load which has the right starting position
#     # Find the end_screen position
#     connector_load = Trajectory_ant(size=file1.size, shape=file1.shape,
#                                     old_filename=file1.VideoChain[-1] + '_CONNECTOR_' + file2.VideoChain[0],
#                                     free=file1.free, fps=file1.fps,
#                                     winner=False,
#                                     )
#     discont = np.pi / periodicity[file1.shape]
#     if con_frames is None:
#         file2.angle = np.unwrap(np.hstack([file1.angle[-1], file2.angle]), discont=discont)[1:]
#         velocity1 = velocity(file1.position, file1.angle, file1.fps, file1.size, file1.shape, 1, 'x', 'y')
#         velocity2 = velocity(file2.position, file2.angle, file2.fps, file2.size, file2.shape, 1, 'x', 'y')
#         dx = file1.position[-1][0] - file2.position[1][0]
#         dy = file1.position[-1][1] - file2.position[1][1]
#         con_vel = max(0.01, np.mean(np.hstack([velocity1[:, -200:-1], velocity2[:, 1:200]])))
#         con_frames = int(np.sqrt(dx ** 2 + dy ** 2) / con_vel)
#
#     connector_load.angle = connector_load.frames = np.ndarray([con_frames])
#     connector_load.angle = np.linspace(file1.angle[-1], file2.angle[0], num=con_frames)
#
#     connector_load.position = np.ndarray([con_frames, 2])
#     connector_load.position[:, 0] = np.linspace(file1.position[-1][0], file2.position[1][0], num=con_frames)
#     connector_load.position[:, 1] = np.linspace(file1.position[-1][1], file2.position[1][1], num=con_frames)
#
#     # all the other stuff
#     connector_load.frames = np.int0(np.linspace(1, con_frames, num=con_frames))
#     return connector_load

#
# def SmoothConnector(file1, file2):
#     """
#     We use this function, when we want to smoothly connect two datasets, where there was an pause between the tracking
#     file1: trajectory1
#     file2: trajectory2
#     con_frames: Number of frames to interpolate over
#     """
#
#     connector_load = run_dstar(size=file1.size,
#                           shape=file1.shape,
#                           solver=file1.solver,
#                           sensing_radius=100,
#                           dil_radius=0,
#                           filename='shortest_path',
#                           starting_point=[file1.position[-1][0], file1.position[-1][1], file1.angle[-1]],
#                           ending_point=[file2.position[0][0], file2.position[0][1], file2.angle[0]],
#                           )
#     return connector_load


def PostTracking_Manipulations_shell(filename):
    def FalseTracking_Smooth():
        x = get(filename)
        if not (hasattr(x, 'falseTracking')):
            x.falseTracking = []

        x.save()

        x.position = np.transpose(
            np.vstack([gaussian_filter(x.position[:, 0], sigma=51), gaussian_filter(x.position[:, 1], sigma=51)]))

        # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
        x.angle = gaussian_filter(x.angle, sigma=51)
        # all the other stuff
        x.save()

    def CutFalseTracking_Free():
        x = get(filename)
        breakpoint()
        if not (hasattr(x, 'falseTracking')):
            x.falseTracking = []
        print(str(x.falseTracking))
        if bool(int(input('Cut off the end_screen '))):
            frame2 = int(input('EndFrame '))
            frame1 = x.frames[0]
        if bool(int(input('Cut off the start '))):
            frame1 = int(input('StartFrame '))
            frame2 = x.frames[-1]
        x.save()

        if not (hasattr(x, 'free')):
            print('Why no free attribute?')
        index1 = np.where(x.frames == frame1)[0][0]
        index2 = np.where(x.frames == frame2)[0][0]

        x.position = x.position[index1:index2, :]

        # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
        x.angle = x.angle[index1:index2]
        x.frames = x.frames[index1:index2]
        x.contact = x.contact[index1:index2]
        breakpoint()
        # all the other stuff
        x.save()
        print(x)

    print('0 = no corrections')
    print('1 = Cut False tracking (if there are issues in the beginning or the end_screen in the free motion)')
    print('3 = Smooth out the trajectory_inheritance')
    print('Connecting to another movie requieres x + y')
    manipulation = int(input('Want to correct something??  '))
    while manipulation != 0:
        if manipulation == 1:
            CutFalseTracking_Free()

        if manipulation == 3:
            FalseTracking_Smooth()

        print('\n 0 = no corrections')
        print('1 = Cut False tracking (if there are issues in the beginning or the end_screen in the free motion)')
        print('3 = Smooth out the trajectory_inheritance')
        print('Connecting to another movie requires x + y')
        manipulation = int(input('Want to correct something else??'))
