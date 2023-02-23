import os
from Directories import SaverDirectories
import pandas as pd

# create a dataframe which has columns: filename, size, solver, shape, fps, winner, position, angle, frames
df_gillespie = pd.DataFrame(columns=['filename', 'size', 'solver', 'shape', 'fps', 'winner'])

for size in ['L', 'M', 'S']:
    folder = 'maze_simulations_' + size + '-SPT_sameKon'
    filenames = os.listdir(os.path.join(SaverDirectories['gillespie'], folder))

    # append the filenames to the dataframe
    df_gillespie = df_gillespie.append(
        pd.DataFrame({'filename': filenames, 'size': size, 'solver': 'gillespie', 'shape': 'SPT',
                      'fps': None, 'winner': None,
                      'maze dimensions': 'MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx',
                      'load dimensions': 'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx'
                      }))

dfs_gillespie = {k: v for k, v in df_gillespie.groupby('size')}

DEBUG = 1
