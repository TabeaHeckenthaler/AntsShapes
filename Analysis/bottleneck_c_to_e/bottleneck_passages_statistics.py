import pandas as pd
from matplotlib import pyplot as plt


def delete_double():
    folder = 'results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_messed_up.xlsx',
                                                index_col=0)
    # go through groups with equal 'filename'
    for filename, group in bottleneck_passing_attempts.groupby('filename'):
        # find all those experiments, where start and end were wihtin another start and end frame
        # and delete them
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue
                if group.iloc[i]['start'] >= group.iloc[j]['start'] >= group.iloc[i]['end']:
                    bottleneck_passing_attempts.drop(group.index[j], inplace=True)
                    print(group.index[i])

    # save the new dataframe
    # bottleneck_passing_attempts.to_excel(folder + 'c_g_bottleneck_phasespace\\bottleneck_passing_attempts_messed_up.xlsx')

    DEBUG = 1


def passage_statistics():
    folder = 'results\\percentage_around_corner\\'
    bottleneck_passing_attempts = pd.read_excel(folder + 'c_g_bottleneck_phasespace\\'
                                                         'bottleneck_passing_attempts_messed_up.xlsx',
                                                index_col=0)

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
    plt.savefig(folder + 'c_g_bottleneck_phasespace\\' + 'passage_percentage.png')
    DEBUG = 1


if __name__ == '__main__':
    delete_double()
    passage_statistics()
