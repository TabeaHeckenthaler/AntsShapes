import json
from Directories import network_dir
from os import path
from itertools import groupby
from DataFrame.import_excel_dfs import df_ant_excluded, dfs_human

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

find_ss = lambda x: [''.join(ii[0]) for ii in groupby([tuple(label) for label in x])]
make_numbers = lambda x: [numbering[s] for s in x]
numbering = {'ab': 4, 'ac': 5, 'b': 3, 'be': 2, 'b1': 0, 'b2': 1,
             'c': 6, 'cg': 7, 'e': 8, 'eb': 9, 'eg': 10, 'f': 11, 'h': 12, 'i': 13}


size = 'XL'
df_ant = df_ant_excluded[df_ant_excluded['size'] == size]
df_ant.loc[:, 'ts'] = df_ant['filename'].apply(lambda x: time_series_dict[x])
df_ant.loc[:, 'ss'] = df_ant['ts'].apply(find_ss)
df_ant.loc[:, 'ss_numbers'] = df_ant['ss'].apply(make_numbers)
df_ant.loc[:, 'ss_numbers'].to_csv('ant_XL.txt', index=False, header=False)

df_human = dfs_human['Large NC']
df_human.loc[:, 'ts'] = df_human['filename'].apply(lambda x: time_series_dict[x])
df_human.loc[:, 'ss'] = df_human['ts'].apply(find_ss)
df_human.loc[:, 'ss_numbers'] = df_human['ss'].apply(make_numbers)
df_human.loc[:, 'ss_numbers'].to_csv('human_L_no_communication.txt', index=False, header=False)

DEBUG = False