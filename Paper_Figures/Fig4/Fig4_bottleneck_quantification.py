import pandas as pd
from matplotlib import pyplot as plt
from Directories import home


plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})


def passage_statistics():
    # split into size groups
    size_groups = bottleneck_passing_attempts.groupby('size')

    # find percentage of winner in every size group
    percentage_winner = {}
    for size, group in size_groups:
        percentage_winner[size] = group['winner'].sum() / len(group)

    # plot the percentage of winner in every size group
    # sort the dictionary by ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)']
    percentage_winner = {size: percentage_winner[size] for size in sizes}

    plt.bar(percentage_winner.keys(), percentage_winner.values())
    plt.ylabel('percentage of winner')
    plt.xlabel('size')
    plt.tight_layout()

    # save as 'passage_percentage'
    plt.savefig('Fig4_bottleneck_passage_percentage.png')
    plt.savefig('Fig4_bottleneck_passage_percentage.svg')
    plt.savefig('Fig4_bottleneck_passage_percentage.eps')
    DEBUG = 1


if __name__ == '__main__':
    folder = home + '\\Analysis\\bottleneck_c_to_e\\results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_messed_up.xlsx',
                                                index_col=0)
    passage_statistics()

