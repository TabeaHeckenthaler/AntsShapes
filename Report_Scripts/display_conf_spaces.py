from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.get import get
# import mayavi
from Setup.Maze import Maze

scale_factor = {'XL': 0.1, 'L': 0.08, 'M': 0.05, 'S': 0.03}


def draw_in_cs(size, filenames: list):
    ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, name=size + '_' + shape + '_' + filenames[0],
                          geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
    ps.load_space()
    ps.visualize_space(reduction=1)

    for filename in filenames:
        x = get(filename)
        angle_modulo = x.angle % (2 * np.pi)

        start, end = [0, -1]
        # start, end = [13600, -1]
        ps.draw(x.position[0:1], angle_modulo[0:1], scale_factor=scale_factor[size] * 5, color=(0, 0, 0))
        ps.draw(x.position[-2:-1], angle_modulo[-2:-1], scale_factor=scale_factor[size] * 5, color=(0, 0, 0))
        ps.draw(x.position[start:end], angle_modulo[start:end], scale_factor=scale_factor[size], color=(1, 0, 0))
        DEBUG = 1
    mayavi.mlab.view(-90, 90)
    mayavi.mlab.move(right=0, forward=1)


if __name__ == '__main__':
    from Analysis.Efficiency.PathLength import path_length_dict
    from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict

    # filenames = myDataFrame[(myDataFrame['size'] == size) & (myDataFrame['shape'] == shape)]['filename'].head(5)
    # filenames = myDataFrame[(myDataFrame['size'] == 'M') & (myDataFrame['solver'] == 'pheidole')]['filename'].head(10)

    # shape, solver = 'SPT', 'pheidole'
    #
    # size_filenames = {
    #     'XL': ['XL_SPT_5240010_XLSpecialT_1 (part 1)'],
    #     'L': ['L_SPT_5230011_LSpecialT_1 (part 1)'],
    #     'M': ['M_SPT_5210003_MSpecialT_1 (part 1)', 'M_SPT_5210005_MSpecialT_1 (part 1)'],
    #     'S': ['S_SPT_5160003_SSpecialT_1_ants (part 1)',
    #           'S_SPT_5160005_SSpecialT_1_ants (part 1)']
    # }

    shape, solver = 'SPT', 'ant'

    size_filenames = {
        'XL': ['XL_SPT_4640012_XLSpecialT_1_ants'],
        'L': ['L_SPT_4660008_LSpecialT_1_ants'],
        'M': ['M_SPT_4700014_MSpecialT_1_ants (part 1)'],
        'S': ['S_SPT_5100010_SSpecialT_1_ants (part 1)']
    }

    for size, filenames in size_filenames.items():
        for filename in filenames:
            x = get(filename)
            print(x.size, Maze(x).average_radius())
            # draw_in_cs(size, [filename])
            # print(size, path_length_dict[filenames[0]] / minimal_path_length_dict[filenames[0]])

        DEBUG = 1
