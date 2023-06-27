from Directories import home, SaverDirectories, mini_work_dir, work_dir, df_dir, data_home
import os
from os import path
import pickle
import numpy as np
from trajectory_inheritance.trajectory import Trajectory
from trajectory_inheritance.trajectory_ant import Trajectory_ant
from trajectory_inheritance.trajectory_human import Trajectory_human
from trajectory_inheritance.trajectory_gillespie import Trajectory_gillespie
# from trajectory_inheritance.trajectory_ps_simulation import Trajectory_ps_simulation
import pandas as pd


def get(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be un-pickled
    :return: trajectory object
    """
    if filename.startswith('sim'):
        size = filename.split('_')[1]
        address = path.join(SaverDirectories['gillespie'], '2023_06_27', size, filename)
        with open(address, 'rb') as f:
            file = pickle.load(f)

        shape, size, solver, filename, fps, position, angle, linVel, angVel, frames, winner = file
        # position = position * {'XL': 4, 'L': 2, 'M': 1, 'S': 1/2}[size]
        x = Trajectory_gillespie(shape=shape, size=size, filename=address.split('\\')[-1],
                                 position=position, angle=angle % (2 * np.pi), frames=frames, winner=winner, fps=fps)
        return x

    # this is on labs network
    for root, dirs, files in os.walk(mini_work_dir):
        for dir in dirs:
            if filename in os.listdir(path.join(root, dir)):
                address = path.join(root, dir, filename)

                with open(address, 'rb') as f:
                    file = pickle.load(f)

                if 'Ant_Trajectories' in address:
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    df = pd.read_excel(df_dir)
                    x = Trajectory_ant(size=size, solver=solver, shape=shape, filename=filename, fps=fps,
                                       winner=winner, position=position, angle=angle, frames=frames,
                                       VideoChain=eval(df[(df['filename'] == filename)]['VideoChain'].iloc[0]),
                                       tracked_frames=eval(df[(df['filename'] == filename)]['tracked_frames'].iloc[0]), )
                    return x

                if 'Human_Trajectories' in address:
                    shape, size, solver, filename, fps, position, angle, frames, winner = file

                    df = pd.read_excel(df_dir)

                    if filename not in df['filename'].values:
                        g = {'filename': filename, 'solver': 'human', 'size': size, 'shape': shape, 'winner': winner,
                             'fps': fps, 'communication': None,
                             'length unit': 'm', 'initial condition': 'back', 'force meter': False,
                             'maze dimensions': 'MazeDimensions_human.xlsx',
                             'load dimensions': 'LoadDimensions_human.xlsx', 'time [s]': None,
                             'comment': '', 'VideoChain': None,
                             'tracked_frames': None, 'free': 0}
                        df2 = pd.DataFrame([g])
                        # df = pd.read_excel(df_dir, usecols=['filename', 'VideoChain', 'tracked_frames'])
                        # df2 = df.append(pd.Series(g), ignore_index=True)
                        df_dir2 = path.join(data_home, 'DataFrame', 'data_frame_add_stuff.xlsx')
                        df2.to_excel(df_dir2)

                        # open excel file and add the new entry
                        os.startfile(df_dir2)
                        input('Add entry to excel file and press enter to continue...')
                        df2 = pd.read_excel(df_dir2)
                        if len(df2['tracked_frames']) > 2:
                            raise Exception('More than 2 tracked frames in excel file.')
                        f = eval(df2['tracked_frames'].values[0])
                        df2['time [s]'] = (f[1] - f[0]) / df2['fps']

                        # concat df to df2 with new index
                        df = pd.concat([df, df2])
                        # drop unnamed column
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        df.reset_index(inplace=True)
                        df.to_json(df_dir)
                        df.to_excel('\\\\phys-guru-cs\\ants\\Tabea\\PyCharm_Data\\AntsShapes\\DataFrame\\data_frame.xlsx')


                    if filename in df['filename'].values:
                        x = Trajectory_human(size=size, shape=shape, filename=filename, fps=fps,
                                             winner=winner, position=position, angle=angle, frames=frames,
                                             VideoChain=df[(df['filename'] == filename)]['VideoChain'].iloc[0],
                                             tracked_frames=df[(df['filename'] == filename)]['tracked_frames'].iloc[0])
                    else:
                        print('No entry in df for this file.')
                        x = Trajectory_human(size=size, shape=shape, filename=filename, fps=fps,
                                             winner=winner, position=position, angle=angle, frames=frames,
                                             VideoChain=None,
                                             tracked_frames=None)
                    return x

                if 'Pheidole_Trajectories' in address:
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    df = pd.read_excel(df_dir)
                    x = Trajectory_ant(size=size, solver=solver, shape=shape, filename=filename, fps=fps,
                                       winner=winner, position=position, angle=angle, frames=frames,
                                       VideoChain=df[(df['filename'] == filename)]['VideoChain'].iloc[0],
                                       tracked_frames=df[(df['filename'] == filename)]['tracked_frames'].iloc[0], )
                    return x

                if 'PS_simulation_Trajectories' in address:
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    x = Trajectory(shape=shape, size=size, solver=solver, filename=filename, position=position,
                                   angle=angle, frames=frames, winner=winner)
                    return x

                if 'Gillespie' in address:
                    shape, size, solver, filename, fps, position, angle, linVel, angVel, frames, winner = file
                    # position = position * {'XL': 4, 'L': 2, 'M': 1, 'S': 1/2}[size]
                    x = Trajectory_gillespie(shape=shape, size=size, filename=address.split('\\')[-1],
                                             position=position, angle=angle % (2*np.pi), frames=frames, winner=winner, fps=fps)
                    return x

                raise Exception('implement get for: ?')
                # return Trajectory(shape=shape, size=size, solver=solver, filename=filename, position=position,
                #                   angle=angle, frames=frames, winner=winner)

        # for root, dirs, files in os.walk(work_dir):
        # for dir in dirs:
        #     if filename in os.listdir(path.join(root, dir)):
        #         address = path.join(root, dir, filename)
        # with open(address, 'rb') as f:
        #     x = pickle.load(f)
        # return x

    # this is local on your computer
    # if filename.startswith('sim'):
    #     raise Exception('you need to import here...')
    #     from trajectory_inheritance.trajectory_gillespie import TrajectoryGillespie
    #     size, shape, solver = 'M', 'SPT', 'gillespie'
    #     with open(os.path.join(SaverDirectories['gillespie'], filename), 'rb') as pickle_file:
    #         data = pickle.load(pickle_file)
    #
    #     x = TrajectoryGillespie(size=size, shape=shape, filename=filename, fps=data.fps, winner=data.winner,
    #                             number_of_attached_ants=data.nAtt)
    #
    #     if len(np.where(np.sum(data.position, axis=1) == 0)[0]):
    #         last_frame = np.where(np.sum(data.position, axis=1) == 0)[0][0]
    #     else:
    #         last_frame = -1
    #     x.position = data.position[:last_frame]
    #     x.angle = data.angle[:last_frame]
    #     x.frames = np.array([i for i in range(last_frame)])
    #     return x
    #
    # local_address = path.join(home, 'trajectory_inheritance', 'trajectories_local')
    # if filename in os.listdir(local_address):
    #     with open(path.join(local_address, filename), 'rb') as f:
    #         print('You are loading ' + filename + ' from local copy.')
    #         x = pickle.load(f)
    #     return x

    else:
        raise ValueError('I cannot find ' + filename)


if __name__ == '__main__':
    filename = 'large_20220527112227_20220527112931'
    x = get(filename)
    x.play()
