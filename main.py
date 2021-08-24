
import GettingStarted_2 as gt
from Classes_Experiment.forces import force_attachment_positions

from trajectory import Get
from Classes_Experiment.humans import Humans
from Setup.Maze import Maze
from Setup.Load import Load
from Classes_Experiment.forces import force_in_frame

def main():


    ''' Display a experiment '''
    # names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
    solver = 'human'
    x = Get('medium_20201221111935_20201221112858', solver)
    x.participants = Humans(x)
    # x.play(forces=[participants_force_arrows])
    # press Esc to stop the display

    ''' Find contact points '''
    contact = []
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
    my_load = Load(my_maze, position=x.position[0])

    # gt.forces_check_func('force_check\\150521.TXT', 'force_check\\force_detector_check_main.png')
    # gt.second_method_graphes(x, force_treshhold=4)
    # measurements_calibration1 = [-1, 3.9, 7.4, 11.3, 11.3, -1, 16.5, 6.3, 2.1, 5.3, 9.1, 11.1, 4.3, 5.6, 2.9, 4.3, 2.2,
    #             1, -2, 10, 6.3, 6, 12.1, 5.4, 6.5, 6.5]  # -1 means broken, -2 means a short pulse
    # gt.forces_check_func('calibration_exp.TXT', 'force_detector_check5.png', measurements_calibration1)

    forces_in_frames = [force_in_frame(x, i) for i in range(len(x.frames))]
    angles = x.angle

    tourqes = [ gt.torque_in_load(my_load, x, forces_in_frames[i], angles[i]) for i in range(len(forces_in_frames))]
    print("SHABAT HAYOM")

if __name__ == "__main__":
    main()