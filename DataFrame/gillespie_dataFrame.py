import os
from Directories import home, SaverDirectories
import pandas as pd
from os import path
import json
from Analysis.PathPy.Path import *
from trajectory_inheritance.get import get

# date = '2023_06_27'
date = 'SimTrjs_RemoveAntsNearWall=False'
# date = 'SimTrjs_RemoveAntsNearWall=True'


def create_new_df():
    df_gillespie = pd.DataFrame(columns=['filename', 'size', 'solver', 'shape', 'fps', 'winner', 'maze dimensions',
                                         'load dimensions'])

    for size in ['XL', 'L', 'M', 'S', 'XS']:
        folder = date + '\\' + size + '\\'
        filenames = os.listdir(os.path.join(SaverDirectories['gillespie'], folder))

        # append the filenames to the dataframe
        df_gillespie = df_gillespie.append(
            pd.DataFrame({'filename': filenames, 'size': size, 'solver': 'gillespie', 'shape': 'SPT',
                          'fps': None, 'winner': True,
                          'maze dimensions': 'MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx',
                          'load dimensions': 'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx'
                          }))
    df_gillespie.to_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')


# create_new_df()
df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')

# append column called 'winner', which is all True
df_gillespie['winner'] = True
dfs_gillespie = {k: v for k, v in df_gillespie.groupby('size')}

ts_directory = home + '\\Gillespie\\' + date + '_sim_time_series.json'

if not os.path.exists(path.join(home, ts_directory)):
    print('Creating new time series dict')

    time_series_dict, state_series_dict = Path.create_dicts(pd.concat(dfs_gillespie), ConfigSpace_SelectedStates,
                                                            dictio_ts=None, dictio_ss=None)

    with open(ts_directory, 'w') as json_file:
        json.dump(time_series_dict, json_file)
        json_file.close()
    Path.plot_paths(zip(['gillespie'], [dfs_gillespie]), time_series_dict, fig_name_add=date)

with open(ts_directory, 'r') as json_file:
    time_series_sim_dict = json.load(json_file)
    json_file.close()


if __name__ == '__main__':

    test = ['sim_XL_2023-07-09_01-52-02.101New',
            'sim_M_20230625-005308New',
            'sim_XS_20230625-193456New',
            'sim_L_20230626-072928New']

    for t in test:
        x = get(t)  # these trajectories are properly scaled, meaning that S is actually 2 times smaller than M etc.
        print(x.position[-1])
        x.play(step=4)
        DEBUG = 1
