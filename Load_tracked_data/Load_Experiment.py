# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from PhysicsEngine.mainGame import mainGame


def Load_Experiment(solver, filename, falseTracking, winner, x_error, y_error, angle_error, fps, free, *args,
                    size=None, shape=None, **kwargs):
    if solver == 'ant':
        from trajectory_inheritance.trajectory_ant import Trajectory_ant
        x = Trajectory_ant(size=size, shape=shape, old_filename=filename, free=free, winner=winner,
                           fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                           falseTracking=[falseTracking])
        if x.free:
            x.RunNum = int(input('What is the RunNumber?'))

    if solver == 'human':
        from trajectory_inheritance.trajectory_human import Trajectory_human
        x = Trajectory_human(size=size, shape=shape, filename=filename, winner=winner,
                             fps=fps, x_error=x_error, y_error=y_error, angle_error=angle_error,
                             falseTracking=[falseTracking])

    else:
        print('Not a valid solver')
        x = None

    x.matlab_loading(filename)  # this is already after we added all the errors...

    if 'frames' in kwargs:
        frames = kwargs['frames']
    else:
        frames = [x.frames[0], x.frames[-1]]
    if len(frames) == 1:
        frames.append(frames[0] + 2)
    f1, f2 = int(frames[0]) - int(x.frames[0]), int(frames[1]) - int(x.frames[0]) + 1
    x.position, x.angle, x.frames = x.position[f1:f2, :], x.angle[f1:f2], x.frames[f1:f2]

    x = mainGame(x, *args, **kwargs)
    return x


if __name__ == '__main__':
    filename = 'large_20210419121802_20210419122542'
    x = Load_Experiment('human', filename, [], True, 0, 0, 0, 30, False, size='L', shape='SPT', )
    x.play()
