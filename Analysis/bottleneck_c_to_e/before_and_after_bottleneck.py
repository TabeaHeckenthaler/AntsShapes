import pandas as pd
from matplotlib import pyplot as plt
from DataFrame.import_excel_dfs import df_ant
from trajectory_inheritance.get import get


df_ant['size'][(df_ant['average Carrier Number'] == 1) & (df_ant['size'] == 'S')] = 'Single (1)'
df_ant['size'][(df_ant['average Carrier Number'] > 1) & (df_ant['size'] == 'S')] = 'S (> 1)'

if __name__ == '__main__':

    folder = 'results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_messed_up.xlsx',
                                                index_col=0)
    DEBUg = 1

    data = pd.DataFrame(columns=['filename', 'time_before_reaching_bottleneck', 'winner', 'size'])
    data['filename'] = df_ant['filename']
    data['winner'] = df_ant['winner']
    data['size'] = df_ant['size']

    # keep only one filename with the smallest value in 'start'
    bottleneck_passing_attempts = bottleneck_passing_attempts.groupby('filename')
    for filename, group in bottleneck_passing_attempts:
        data.loc[data['filename'] == filename, 'time_before_reaching_bottleneck'] = \
            group['start'].min() / df_ant[df_ant['filename'] == filename]['fps'] / 60

    # where ever there is nan in 'time_before_reaching_bottleneck', replace it with the number of frames/fps
    data['time_before_reaching_bottleneck'].fillna(df_ant['time [s]'] / 60, inplace=True)

    # draw a histogram of 'time_before reaching_bottleneck' for every size on a different axis
    order = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    size_groups = data.groupby('size', )
    fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
    for size, ax in zip(['XL', 'L', 'M', 'S (> 1)', 'Single (1)'], axs):
        group = data[data['size'] == size]
        ax.hist(group['time_before_reaching_bottleneck'], bins=10, density=True, label=size)
        ax.set_title(size)

    axs[2].set_xlabel('time before reaching bottleneck [min]')
    axs[0].set_ylabel('probability density')
    plt.tight_layout()
    plt.savefig(folder + 'c_g_bottleneck_phasespace\\' + 'time_before_reaching_bottleneck.png')
    DEBUG = 1