import numpy as np

import StartedScripts.GettingStarted_2 as gt
from Classes_Experiment.forces import force_attachment_positions
import glob
import os
import matplotlib.pyplot as plt

from trajectory import Get
from Classes_Experiment.humans import Humans
from Setup.Maze import Maze
from Setup.Load import Load
from Classes_Experiment.forces import force_in_frame
from Classes_Experiment.humans import force_filename,force_directory

''''''

def main():
    ''' Display a experiment '''
    # names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
    solver = 'human'
    x = Get('medium_20201223105429_20201223110802', solver)
    x.participants = Humans(x)
    # x.play(forces=[participants_force_arrows])
    # press Esc to stop the display

    ''' Find contact points '''
    contact = []
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
    my_load = Load(my_maze, position=x.position[0])

    directory_of_txt = force_directory(x) + '\\' + force_filename(x)

    # gt.forces_check_func('force_check\\150521.TXT', 'force_check\\force_detector_check_main.png')
    # gt.second_method_graphes(x, force_treshhold=4)
    # measurements_calibration1 = [-1, 3.9, 7.4, 11.3, 11.3, -1, 16.5, 6.3, 2.1, 5.3, 9.1, 11.1, 4.3, 5.6, 2.9, 4.3, 2.2,
    #             1, -2, 10, 6.3, 6, 12.1, 5.4, 6.5, 6.5]  # -1 means broken, -2 means a short pulse
    # gt.forces_check_func('calibration_exp.TXT', 'force_detector_check5.png', measurements_calibration1)

    forces_in_frames = [force_in_frame(x, i) for i in range(len(x.frames))]
    angles = x.angle
    ratio = (0.453592)

    tourqes = [gt.torque_in_load(my_load, x, forces_in_frames[j], angles[j]) for j in range(len(forces_in_frames))]
    ang_vel = [gt.numeral_angular_velocity(x, i) for i in range(len(x.angle))]

    for number in range(9):
        gt.single_force_check_func(directory_of_txt, str.split(force_filename(x),'.')[0] + 'png', number,
                                   action='open', size='M', ratio=ratio)

    # measurements_calibration = {'153151': [],
    #                             '153632': [-1, 17.4, 14.1, 12.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                                        22.5, 12.1, 31.1, 24.7],
    #                             '155425': [-1, 0, 0, 0, 13.0, -1, 20.1, 22.3, 28.0, 19.2, 23.6, 10.4, 15.4, 0, 0, 0, 0,
    #                                        0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                             '161453': [-1, 0, 0, 0, 13.0, -1, 0, 0, 0, 0, 0, 0, 0, 11.5, 20.7, 14.1, 15.2, 10.1,
    #                                        23.1, 30.6, 13.4, 11.0, 0, 0, 0, 0]}

    # gt.second_method_graphes(x, force_treshhold=2)

    # for filename in glob.glob('force_check_Aug_19\\*.TXT'):
    #     with open(os.path.join(os.getcwd(), filename), 'r') as f:
    #         g = ((filename.split('\\'))[1].split('.'))[0]
    #         gt.forces_check_func(filename,'force_check_Aug_19\\' + g + '.png', action='save',  size = 'L',ratio = ratio,
    #                              measured_forces = measurements_calibration[g])
    #
    # ABC = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # gives an array of the ABC...
    #
    # for filename in glob.glob('force_check_Aug_19\\*.TXT'):
    #     with open(os.path.join(os.getcwd(), filename), 'r') as f:
    #         g = ((filename.split('\\'))[1].split('.'))[0]
    #         for letter in ABC:
    #              gt.single_force_check_func(filename,'force_check_Aug_19\\' + g + '_' +  letter  + '.png',letter,  action='save',
    #                                         size = 'L', ratio = ratio, measured_forces= measurements_calibration[g])

    # for filename in glob.glob('medium check_exp\\*.TXT'):
    #     with open(os.path.join(os.getcwd(), filename), 'r') as f:
    #         g = ((filename.split('\\'))[1].split('.'))[0]
    #         gt.forces_check_func(filename, 'medium check_exp\\' + g + '.png', action='save', size='L', ratio=ratio)
    #     for number in range(9):
    #         gt.single_force_check_func(filename, 'medium check_exp\\' + g + '_' + str(number) + '.png', number,
    #                                    action='open', size='M', ratio=ratio)


if __name__ == "__main__":
    main()
