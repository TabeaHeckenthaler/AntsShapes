import matplotlib.pyplot as plt
import pandas as pd
from DataFrame.import_excel_dfs import df_ant_excluded


# read excel file
df = pd.read_excel('ant_counting_prep_only_carrying.xlsx')

# rename column from 'Unnamed: 0' to 'filename'
df.rename(columns={'Unnamed: 0': 'filename'}, inplace=True)

# in every row, replace the filename with the filename the last two letters
df['filename'] = df['filename'].apply(lambda x: x[:-2])

# merge with df_ant_excluded
df = df.merge(df_ant_excluded[['filename', 'size']], on='filename')
df['all'] = df['outside'] + df['inside']

# group by size
df_grouped = df.groupby('size')

# find mean and std of df_grouped
df_mean = df_grouped['all'].mean()
df_std = df_grouped['all'].std()

# merge mean and std to single df
df_ant_count = pd.DataFrame({'mean': df_mean, 'std': df_std})

df_ant_count = df_ant_count.reindex(['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:])

# save to excel
df_ant_count.to_excel('ant_counting_means_only_carrying.xlsx')

# plot the mean and std
plt.figure()
plt.bar(df_ant_count.index, df_ant_count['mean'], yerr=df_ant_count['std'])
plt.title('Ant count (carrying)')
plt.ylabel('Ant count')
plt.xlabel('Size')
plt.savefig('ant_counting_means_only_carrying.png')


def outside_inside():
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
