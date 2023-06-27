from DataFrame.import_excel_dfs import df_ant_excluded
import pandas as pd
import matplotlib.pyplot as plt
from colors import colors_state
from DataFrame.import_excel_dfs import df_ant_excluded
import json
from os import path
from Directories import network_dir, home
from itertools import groupby
from trajectory_inheritance.get import get
from tqdm import tqdm
import numpy as np


# read excel file
df = pd.read_excel('ant_counting_prep.xlsx')

# rename column from 'Unnamed: 0' to 'filename'
df.rename(columns={'Unnamed: 0': 'filename'}, inplace=True)

# in every row, replace the filename with the filename the last two letters
df['filename'] = df['filename'].apply(lambda x: x[:-2])

# merge with df_ant_excluded
df = df.merge(df_ant_excluded[['filename', 'size']], on='filename')

# group by size
df = df.groupby('size')

df_outside = df['outside'].mean()
df_inside = df['inside'].mean()

# calc std
df_outside_std = df['outside'].std()
df_inside_std = df['inside'].std()


# sort index by list
df_outside = df_outside.reindex(['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:])
df_inside = df_inside.reindex(['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:])

# find number of experiments per size which are not nan
num_experiments = df.count()['outside']


# combine all series df_outside_std, df_inside_std, df_outside, df_inside
# to single df with size as index and columns 'inside' and 'outside'
df_results = pd.DataFrame({'outside': df_outside, 'outside std': df_outside_std, 'inside': df_inside,
                           'inside std': df_inside_std, 'numExp': num_experiments})
df_results = df_results.reindex(['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:])

# save to excel
df_results.to_excel('ant_counting_means.xlsx')

# plot
plt.figure()
plt.bar(df_outside.index, df_outside.values)
plt.title('Outside')

plt.figure()
plt.bar(df_inside.index, df_inside.values)
plt.show()
plt.title('Inside')

DEBUG = 1