from os import path
from Directories import lists_exp_dir
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

df_minimal = pd.read_excel(path.join(lists_exp_dir, 'exp_minimal.xlsx'), engine='openpyxl')
def find_minimal(s):
    return df_minimal[(df_minimal['size'] == s['size']) & (df_minimal['shape'] == 'SPT') & (
            df_minimal['initial condition'] == 'back')].iloc[0]['path length [length unit]']

df_all = pd.read_excel(path.join(lists_exp_dir, 'exp.xlsx'), engine='openpyxl')

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
print('Have you added the human experiment from the 19/02?')

dfs_human = {'Large C': df_exp_human_L_C,
             'Large NC': df_exp_human_L_NC,
             'Medium C': df_exp_human_M_C,
             'Medium NC': df_exp_human_M_NC,
             'Small': df_exp_human_S
             }

df_human = df_all[df_all['solver'] == 'human']
df_human = df_human[df_human['initial condition'] == 'back']

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
