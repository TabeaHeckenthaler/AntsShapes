from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.get import get
import mayavi

scale_factor = {'XL': 0.1, 'M': 0.05}


def draw_in_cs():
    ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, name=size + '_' + shape,
                          geometry=('MazeDimensions_ant.xlsx', 'LoadDimensions_ant.xlsx'))
    ps.load_space()
    ps.visualize_space()

    for filename in filenames:
        x = get(filename)
        angle_modulo = x.angle % (2 * np.pi)
        ps.draw(x.position[0:1], angle_modulo[0:1], scale_factor=scale_factor[size]*5, color=(0, 0, 0))
        ps.draw(x.position[-2:-1], angle_modulo[-2:-1], scale_factor=scale_factor[size]*5, color=(0, 0, 0))
        ps.draw(x.position, angle_modulo, scale_factor=scale_factor[size])
        DEBUG = 1
    mayavi.mlab.view(-90, 90)
    mayavi.mlab.move(right=0, forward=1)


if __name__ == '__main__':
    size, shape, solver = 'M', 'I', 'ant'
    filenames = myDataFrame[(myDataFrame['size'] == size) & (myDataFrame['shape'] == shape)]['filename'].head(5)
    draw_in_cs()
    DEBUG = 1
