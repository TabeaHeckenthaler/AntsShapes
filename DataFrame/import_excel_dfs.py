from os import path
from Directories import lists_exp_dir
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

df_minimal = pd.read_excel(path.join(lists_exp_dir, 'exp_minimal.xlsx'), engine='openpyxl')


def find_minimal_pL(s):
    if s['size'] == 'Small Near':
        size = 'Small Far'
    else:
        size = s['size']
    index = (df_minimal['size'] == size) & (df_minimal['shape'] == 'SPT') & \
            (df_minimal['initial condition'] == 'back') & (df_minimal['maze dimensions'] == s['maze dimensions'])
    # find where index is true
    if len(index[index].index) != 1 & \
            np.unique([df_minimal.iloc[i]['path length [length unit]'] for i in index[index].index]).size > 1:
        raise Exception('index is not unique')

    # print(s['filename'])
    # if s['filename'] == 'small_20201223095702_20201223100014':
    #     DEBUG = 1
    return df_minimal[index].iloc[0]['path length [length unit]']


print('Do NOT use the time [s] column, it is not accurate. Use the frameNum dict instead.')

df_all = pd.read_excel(path.join(lists_exp_dir, 'exp.xlsx'), engine='openpyxl')  # missing new human experiments

file_path_exp_ant_XL_winner = path.join(lists_exp_dir, 'exp_ant_XL_winner.xlsx')
file_path_exp_ant_XL_looser = path.join(lists_exp_dir, 'exp_ant_XL_looser.xlsx')
file_path_exp_ant_L_winner = path.join(lists_exp_dir, 'exp_ant_L_winner.xlsx')
file_path_exp_ant_L_looser = path.join(lists_exp_dir, 'exp_ant_L_looser.xlsx')
file_path_exp_ant_M_winner = path.join(lists_exp_dir, 'exp_ant_M_winner.xlsx')
file_path_exp_ant_M_looser = path.join(lists_exp_dir, 'exp_ant_M_looser.xlsx')
file_path_exp_ant_S_winner = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_winner.xlsx')
file_path_exp_ant_S_looser = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_looser.xlsx')
file_path_exp_ant_Single_winner = path.join(lists_exp_dir, 'exp_ant_Single (1)_looser.xlsx')
file_path_exp_ant_Single_looser = path.join(lists_exp_dir, 'exp_ant_Single (1)_winner.xlsx')

file_path_exp_ant_XL_winner_old = path.join(lists_exp_dir, 'exp_ant_XL_winner_old.xlsx')
file_path_exp_ant_XL_looser_old = path.join(lists_exp_dir, 'exp_ant_XL_looser_old.xlsx')
file_path_exp_ant_L_winner_old = path.join(lists_exp_dir, 'exp_ant_L_winner_old.xlsx')
file_path_exp_ant_L_looser_old = path.join(lists_exp_dir, 'exp_ant_L_looser_old.xlsx')
file_path_exp_ant_M_winner_old = path.join(lists_exp_dir, 'exp_ant_M_winner_old.xlsx')
file_path_exp_ant_M_looser_old = path.join(lists_exp_dir, 'exp_ant_M_looser_old.xlsx')
file_path_exp_ant_S_winner_old = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_winner_old.xlsx')
file_path_exp_ant_S_looser_old = path.join(lists_exp_dir, 'exp_ant_S (more than 1)_looser_old.xlsx')
file_path_exp_ant_Single_winner_old = path.join(lists_exp_dir, 'exp_ant_Single (1)_looser_old.xlsx')
file_path_exp_ant_Single_looser_old = path.join(lists_exp_dir, 'exp_ant_Single (1)_winner_old.xlsx')

df_exp_ant_XL_looser = pd.read_excel(file_path_exp_ant_XL_looser, engine='openpyxl')
df_exp_ant_XL_winner = pd.read_excel(file_path_exp_ant_XL_winner, engine='openpyxl')
df_exp_ant_L_looser = pd.read_excel(file_path_exp_ant_L_looser, engine='openpyxl')
df_exp_ant_L_winner = pd.read_excel(file_path_exp_ant_L_winner, engine='openpyxl')
df_exp_ant_M_looser = pd.read_excel(file_path_exp_ant_M_looser, engine='openpyxl')
df_exp_ant_M_winner = pd.read_excel(file_path_exp_ant_M_winner, engine='openpyxl')
df_exp_ant_S_looser = pd.read_excel(file_path_exp_ant_S_looser, engine='openpyxl')
df_exp_ant_S_winner = pd.read_excel(file_path_exp_ant_S_winner, engine='openpyxl')
df_exp_ant_Single_winner = pd.read_excel(file_path_exp_ant_Single_winner, engine='openpyxl')
df_exp_ant_Single_looser = pd.read_excel(file_path_exp_ant_Single_looser, engine='openpyxl')

df_exp_ant_XL_looser_old = pd.read_excel(file_path_exp_ant_XL_looser_old, engine='openpyxl')
df_exp_ant_XL_winner_old = pd.read_excel(file_path_exp_ant_XL_winner_old, engine='openpyxl')
df_exp_ant_L_looser_old = pd.read_excel(file_path_exp_ant_L_looser_old, engine='openpyxl')
df_exp_ant_L_winner_old = pd.read_excel(file_path_exp_ant_L_winner_old, engine='openpyxl')
df_exp_ant_M_looser_old = pd.read_excel(file_path_exp_ant_M_looser_old, engine='openpyxl')
df_exp_ant_M_winner_old = pd.read_excel(file_path_exp_ant_M_winner_old, engine='openpyxl')
df_exp_ant_S_looser_old = pd.read_excel(file_path_exp_ant_S_looser_old, engine='openpyxl')
df_exp_ant_S_winner_old = pd.read_excel(file_path_exp_ant_S_winner_old, engine='openpyxl')
df_exp_ant_Single_winner_old = pd.read_excel(file_path_exp_ant_Single_winner_old, engine='openpyxl')
df_exp_ant_Single_looser_old = pd.read_excel(file_path_exp_ant_Single_looser_old, engine='openpyxl')

df_ant = df_all[df_all['solver'] == 'ant']
df_ant = df_ant[df_ant['filename'].str.contains('SPT')]
df_ant = df_ant[df_ant['initial condition'] == 'back']
df_ant = df_ant[df_ant['maze dimensions'].isin(['MazeDimensions_new2021_SPT_ant.xlsx'])]

exclude = ['M_SPT_4690001_MSpecialT_1_ants', 'M_SPT_4690011_MSpecialT_1_ants (part 1)',
           'L_SPT_4650012_LSpecialT_1_ants (part 1)', 'L_SPT_4660014_LSpecialT_1_ants',
           'L_SPT_4670008_LSpecialT_1_ants (part 1)', 'L_SPT_4420007_LSpecialT_1_ants (part 1)',
           'L_SPT_4420004_LSpecialT_1_ants', 'L_SPT_4420005_LSpecialT_1_ants (part 1)',
           'L_SPT_4420010_LSpecialT_1_ants (part 1)', 'L_SPT_5030001_LSpecialT_1_ants (part 1)',
           'L_SPT_5030009_LSpecialT_1_ants (part 1)', 'L_SPT_5030006_LSpecialT_1_ants (part 1)',
           'XL_SPT_4640001_XLSpecialT_1_ants (part 1)', 'XL_SPT_5040006_XLSpecialT_1_ants (part 1)',
           'XL_SPT_5040012_XLSpecialT_1_ants', 'XL_SPT_5040003_XLSpecialT_1_ants (part 1)']
# ant density in back room to low

df_ant['size'][(df_ant['average Carrier Number'] == 1) & (df_ant['size'] == 'S')] = 'Single (1)'
df_ant['size'][(df_ant['average Carrier Number'] > 1) & (df_ant['size'] == 'S')] = 'S (> 1)'

df_ant_excluded = df_ant[~df_ant['filename'].isin(exclude)]
# write to excel
# df_ant_excluded.to_excel('ant_experiments.xlsx', index=False)

df_ant_old = df_all[(df_all['solver'] == 'ant') & df_all['filename'].str.contains('SPT') &
                    (df_all['initial condition'] == 'back')]

dfs_ant = {'XL': pd.concat([df_exp_ant_XL_winner, df_exp_ant_XL_looser]),
           'L': pd.concat([df_exp_ant_L_winner, df_exp_ant_L_looser]),
           'M': pd.concat([df_exp_ant_M_winner, df_exp_ant_M_looser]),
           'S (> 1)': pd.concat([df_exp_ant_S_winner, df_exp_ant_S_looser]),
           'Single (1)': pd.concat([df_exp_ant_Single_winner, df_exp_ant_Single_looser]),
           }

dfs_ant_old = {'XL': pd.concat([df_exp_ant_XL_winner_old, df_exp_ant_XL_looser_old]),
               'L': pd.concat([df_exp_ant_L_winner_old, df_exp_ant_L_looser_old]),
               'M': pd.concat([df_exp_ant_M_winner_old, df_exp_ant_M_looser_old]),
               'S (> 1)': pd.concat([df_exp_ant_S_winner_old, df_exp_ant_S_looser_old]),
               'Single (1)': pd.concat([df_exp_ant_Single_winner_old, df_exp_ant_Single_looser_old]),
               }

file_path_exp_human_L_C = path.join(lists_exp_dir, 'exp_human_Large_communication.xlsx')
file_path_exp_human_L_NC = path.join(lists_exp_dir, 'exp_human_Large_non_communication.xlsx')
file_path_exp_human_M_C = path.join(lists_exp_dir, 'exp_human_M (more than 7)_communication.xlsx')
file_path_exp_human_M_NC = path.join(lists_exp_dir, 'exp_human_M (more than 7)_non_communication.xlsx')
file_path_exp_human_S = path.join(lists_exp_dir, 'exp_human_Small_non_communication.xlsx')

df_exp_human_L_C = pd.read_excel(file_path_exp_human_L_C, engine='openpyxl')
df_exp_human_L_NC = pd.read_excel(file_path_exp_human_L_NC, engine='openpyxl')
df_exp_human_M_C = pd.read_excel(file_path_exp_human_M_C, engine='openpyxl')
df_exp_human_M_NC = pd.read_excel(file_path_exp_human_M_NC, engine='openpyxl')
df_exp_human_S = pd.read_excel(file_path_exp_human_S, engine='openpyxl')
# print('Have you added the human experiment from the 13/03?')

dfs_human = {'Large C': df_exp_human_L_C,
             'Large NC': df_exp_human_L_NC,
             'Medium C': df_exp_human_M_C,
             'Medium NC': df_exp_human_M_NC,
             'Small': df_exp_human_S
             }

# concatenate all the dataframes in dfs_human
df_human = pd.concat(dfs_human.values())

# # save to excel
# df_human.to_excel('df_human.xlsx', index=False)

# df_human1 = df_all[df_all['solver'] == 'human']
# df_human1 = df_human1[df_human1['initial condition'] == 'back']

# find difference between filenames in df_human['filename'] and df_human1['filename']

file_path_exp_pheidole_XL_winner = path.join(lists_exp_dir, 'exp_pheidole_XL_winner_.xlsx')
file_path_exp_pheidole_XL_looser = path.join(lists_exp_dir, 'exp_pheidole_XL_looser_.xlsx')
file_path_exp_pheidole_L_winner = path.join(lists_exp_dir, 'exp_pheidole_L_winner_.xlsx')
file_path_exp_pheidole_L_looser = path.join(lists_exp_dir, 'exp_pheidole_L_looser_.xlsx')
file_path_exp_pheidole_M_winner = path.join(lists_exp_dir, 'exp_pheidole_M_winner_.xlsx')
file_path_exp_pheidole_M_looser = path.join(lists_exp_dir, 'exp_pheidole_M_looser_.xlsx')
file_path_exp_pheidole_S_winner = path.join(lists_exp_dir, 'exp_pheidole_S (more than 1)_winner_.xlsx')
file_path_exp_pheidole_S_looser = path.join(lists_exp_dir, 'exp_pheidole_S (more than 1)_looser_.xlsx')

df_exp_pheidole_XL_looser = pd.read_excel(file_path_exp_pheidole_XL_looser, engine='openpyxl')
df_exp_pheidole_XL_winner = pd.read_excel(file_path_exp_pheidole_XL_winner, engine='openpyxl')
df_exp_pheidole_L_looser = pd.read_excel(file_path_exp_pheidole_L_looser, engine='openpyxl')
df_exp_pheidole_L_winner = pd.read_excel(file_path_exp_pheidole_L_winner, engine='openpyxl')
df_exp_pheidole_M_looser = pd.read_excel(file_path_exp_pheidole_M_looser, engine='openpyxl')
df_exp_pheidole_M_winner = pd.read_excel(file_path_exp_pheidole_M_winner, engine='openpyxl')
df_exp_pheidole_S_looser = pd.read_excel(file_path_exp_pheidole_S_looser, engine='openpyxl')
df_exp_pheidole_S_winner = pd.read_excel(file_path_exp_pheidole_S_winner, engine='openpyxl')

df_pheidole = df_all[df_all['solver'] == 'pheidole']
df_pheidole = df_pheidole[df_pheidole['filename'].str.contains('SPT')]
df_pheidole = df_pheidole[df_pheidole['initial condition'] == 'back']
df_pheidole = df_pheidole[df_pheidole['maze dimensions'].isin(['MazeDimensions_new2021_SPT_ant.xlsx'])]

dfs_pheidole = {'XL': pd.concat([df_exp_pheidole_XL_winner, df_exp_pheidole_XL_looser]),
                'L': pd.concat([df_exp_pheidole_L_winner, df_exp_pheidole_L_looser]),
                'M': pd.concat([df_exp_pheidole_M_winner, df_exp_pheidole_M_looser]),
                'S (> 1)': pd.concat([df_exp_pheidole_S_winner, df_exp_pheidole_S_looser]),
                }
df_relevant = pd.concat([df_human, df_pheidole, df_ant], ignore_index=True)
DEBUG = 1
