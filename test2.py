import pandas as pd
from Directories import lists_exp_dir, network_dir
from os import path
from pandas import read_excel
import json
import numpy as np
from trajectory_inheritance.get import get
from Analysis.PathPy.Path import Path
#
# file_path1 = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_looser.xlsx')
# file_path2 = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_winner.xlsx')
#
# df1 = read_excel(file_path1, engine='openpyxl')
# df2 = read_excel(file_path2, engine='openpyxl')
#
# df = pd.concat([df1, df2], ignore_index=True)

# with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
#     time_series_dict = json.load(json_file)
#     json_file.close()
# df['time series'] = df['filename'].map(time_series_dict)

filename = 'S_SPT_5190017_SSpecialT_1_ants'
x = get(filename)
# frames has to be
x.reduce_fps(x.fps)
x.load_participants()
x.play(wait=100)

DEBUG = 1