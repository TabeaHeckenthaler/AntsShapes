from PhaseSpaces import PhaseSpace
import os
import numpy as np
from mayavi import mlab
from Analysis_Functions.GeneralFunctions import graph_dir
from DataFrame.create_dataframe import df
from trajectory import Get
from Directories import data_home

filenames_group = df[['filename', 'solver', 'maze size', 'shape']].groupby(['solver', 'maze size', 'shape'])

for (solver, size, shape), df1 in filenames_group:
    for index in df1.index[::4]:
        if solver != 'humanhand':
            filename = df1['filename'].loc[index]
            x = Get(filename, solver)

            position = np.vstack([position, x.position])
            angle = np.hstack([angle, x.angle])
            position = position[1:]
            angle = angle[1:]

            # x = list(my_bundle)[0]
            # maze, load = Load(Maze(x), x)

            ps = PhaseSpace.PhaseSpace(solver, size, shape,
                                       name=size + '_' + shape)

            data_dir = data_home + 'PhaseSpace\\' + solver

            ps.load_space(path=os.path.join(data_dir, ps.name + ".pkl"))

            # y_range = max(abs(np.min(position[:, 1]) - maze.arena_height/2),
            #               abs(np.max(position[:, 1]) - maze.arena_height/2))
            #
            # ps.trim([[np.min(position[:, 0]), np.max(position[:, 0])],
            #          [maze.arena_height/2 - y_range, maze.arena_height/2 + y_range]])

            fig = ps.visualize_space(ps.name)

            fig = ps.draw_trajectory(fig, position, angle,
                                     scale_factor=0.4,
                                     color=(1, 0, 0))

            mlab.savefig(graph_dir() + os.path.sep + ps.name + '.jpg', magnification=4)
            mlab.close()

