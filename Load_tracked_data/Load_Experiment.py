# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:35:01 2020

@author: tabea
"""
from PhysicsEngine.Box2D_GameLoops import MainGameLoop


def Load_Experiment(solver, old_filename, falseTracking, winner, x_error, y_error, angle_error, fps, free, *args,
                    size=None, shape=None, **kwargs):
    import trajectory
    x = trajectory.Trajectory(old_filename=old_filename, solver=solver, winner=winner,
                              fps=fps, free=free, x_error=x_error, y_error=y_error, angle_error=angle_error,
                              shape=shape, size=size,
                              falseTracking=[falseTracking], **kwargs)

    if x.free:
        x.RunNum = int(input('What is the RunNumber?'))
    x.matlab_loading(old_filename)  # this is already after we added all the errors...

    if 'frames' in kwargs:
        frames = kwargs['frames']
    else:
        frames = [x.frames[0], x.frames[-1]]
    if len(frames) == 1:
        frames.append(frames[0] + 2)
    f1, f2 = int(frames[0]) - int(x.frames[0]), int(frames[1]) - int(x.frames[0]) + 1
    x.position, x.angle, x.frames = x.position[f1:f2, :], x.angle[f1:f2], x.frames[f1:f2]

    if 'L_I_425' in x.filename:
        args = args + ('L_I1', )
    x = MainGameLoop(x, *args, **kwargs)
    return x


if __name__ == '__main__':
    old_filename = 'large_20210419121802_20210419122542'
    x = Load_Experiment('human', old_filename, [], True, 0, 0, 0, 30, False, size='L', shape='SPT', )
    x.play()

    pass