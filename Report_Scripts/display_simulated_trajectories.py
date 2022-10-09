from Directories import SaverDirectories, df_sim_dir
import os
from trajectory_inheritance.trajectory_gillespie import TrajectoryGillespie
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from DataFrame.dataFrame import DataFrame
from trajectory_inheritance.get import get

filenames = os.listdir(SaverDirectories['gillespie'])
size, shape, solver = 'M', 'SPT', 'gillespie'


def draw_in_cs():
    ps = ConfigSpace_Maze(solver=solver, size=size, shape=shape, name=size + '_' + shape,
                          geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
    ps.load_space()
    ps.visualize_space()

    for filename in filenames[:]:
        x = get(filename)
        # angle_modulo = x.angle % (2 * np.pi)
        # ps.draw(x.position[0:1], angle_modulo[0:1], scale_factor=0.5, color=(0, 0, 0))
        # ps.draw(x.position[-2:-1], angle_modulo[0:1], scale_factor=0.5, color=(0, 0, 0))
        # ps.draw(x.position, angle_modulo, scale_factor=0.05)


def create_df():
    df = DataFrame.create({'gillespie': filenames})
    carrierNumbers = {}

    for filename in filenames[:]:
        x = get(filename)
        carrierNumbers.update({filename: x.averageCarrierNumber()})

    df['counted carrier number'] = df['filename'].map(carrierNumbers)
    myDataFrame = DataFrame(df)
    myDataFrame.save(df_sim_dir)


if __name__ == '__main__':
    myDataFrame = DataFrame(pd.read_json(df_sim_dir))
    DEBUG = 1