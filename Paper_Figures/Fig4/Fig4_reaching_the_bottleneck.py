import pandas as pd
from matplotlib import pyplot as plt
from Directories import home, network_dir
import json
from os import path
import numpy as np
from tqdm import tqdm
from DataFrame.import_excel_dfs import df_ant_excluded

ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:]
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})


def plot_statistics():
    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54 * 1.5, 5))

    # load from json
    with open(folder + 'c_g_bottleneck_phasespace\\Num_entrances.json', 'r') as fp:
        Num_entrances = json.load(fp)

    with open(folder + 'c_g_bottleneck_phasespace\\time_in_bottleneck.json', 'r') as fp:
        time_in_bottleneck = json.load(fp)

    # plot the mean
    for size in ant_sizes:
        n = Num_entrances[size]
        axs[0].bar(size, np.mean(n), yerr=np.std(n)/np.sqrt(len(n)), capsize=5, color='lightskyblue')
        # t = time_in_bottleneck[size]
        # axs[1].bar(size, np.mean(t), yerr=np.std(t)/np.sqrt(len(t)), capsize=5, color='lightskyblue')

    axs[0].set_ylabel('Number of entrances to bottleneck')
    # axs[1].set_ylabel('Fraction of time in bottleneck')
    DEBUG = 1

    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts.xlsx',
                                                index_col=0)
    size_groups = bottleneck_passing_attempts.groupby('size')

    # find percentage of winner in every size group
    percentage_winner = {}
    std = {}
    for size in ant_sizes:
        group = size_groups.get_group(size)
        percentage_winner[size] = group['winner'].sum() / len(group)
        std[size] = np.sqrt(percentage_winner[size] * (1 - percentage_winner[size]) / len(group))

    # plot the percentage of winner in every size group
    # sort the dictionary by ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    percentage_winner = {size: percentage_winner[size] for size in ant_sizes}

    axs[1].bar(ant_sizes, [percentage_winner[size] for size in ant_sizes],
               yerr=[std[size] for size in ant_sizes], capsize=5,
               color='lightskyblue')
    axs[1].set_ylabel('bottleneck passage probability')
    axs[1].set_xlabel('size')
    plt.tight_layout()


if __name__ == '__main__':
    folder = home + '\\Analysis\\bottleneck_c_to_e\\results\\percentage_around_corner\\'
    plot_statistics()
    plt.savefig('bottleneck_statistics.png', dpi=300)
    plt.savefig('bottleneck_statistics.svg', dpi=300)
    plt.savefig('bottleneck_statistics.eps', dpi=300)
    plt.close()