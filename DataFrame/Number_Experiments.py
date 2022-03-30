from DataFrame.plot_dataframe import save_fig
from matplotlib import pyplot as plt
from trajectory_inheritance.trajectory import sizes
from DataFrame.dataFrame import myDataFrame as df


def how_many_experiments_SPT(df, initial_cond='back'):
    """
    plot how much data you already have
    :param df:, data frame with all the experiments
    :param initial_cond: which experiments to include
    :param shapes: which shapes to include
    """
    shapes = ['SPT']
    fig, axs = plt.subplots(1, 2, sharey=True)
    df = df[(df['initial condition'] == initial_cond) & (df['shape'].isin(shapes))]

    ant_df = df.loc[df.groupby('solver').groups['ant']]
    ant_dict = {'S': [], 'M': [], 'L': [], 'XL': []}
    ant_dict.update(ant_df.groupby('size').groups)
    sizes_here = [s for s in sizes['ant'] if s in ant_dict.keys()]

    axs[0].bar(range(len(ant_dict)),
               [len(ant_dict[size]) for size in sizes_here],
               label='no communication', color='k')
    axs[0].set_xticks(range(len(ant_dict)))
    axs[0].set_xticklabels(sizes_here)
    axs[0].set_title('ants')
    axs[0].set_ylabel('number of experiments')
    axs[0].set_xlabel('maze size')
    axs[0].set_ylim(0, 60)

    human_df_all = df.loc[df.groupby('solver').groups['human']]

    def human_dict(human_df):
        return dict(S=human_df[human_df['size'].isin(['Small Far', 'Small Near'])].index,
                    couples=human_df[human_df['average Carrier Number'] == 2].index,
                    M=human_df[(human_df['size'] == 'Medium') & (human_df['average Carrier Number'] != 2)].index,
                    L=human_df[human_df['size'] == 'Large'].index)

    human_sizes = ['S', 'couples', 'M', 'L']
    human_dict_Com = human_dict(human_df_all[human_df_all['communication']])
    human_dict_NoCom = human_dict(human_df_all[~human_df_all['communication']])
    heightNoCom = [len(human_dict_NoCom[size]) for size in human_sizes]
    heightCom = [len(human_dict_Com[size]) for size in human_sizes]

    axs[1].bar(range(len(human_dict_Com)), heightNoCom, label='no communication', color='r')
    axs[1].bar(range(len(human_dict_Com)), heightCom, bottom=heightNoCom, label='communication', color='blue')
    axs[1].set_xticks(range(len(human_dict_Com)))
    axs[1].set_xticklabels(human_sizes)
    axs[1].set_title('humans')
    axs[1].set_xlabel('maze size')
    axs[1].legend()

    fig.suptitle('initial condition: ' + initial_cond)
    save_fig(fig, 'how_many_experiments_' + initial_cond + '_' + ''.join(shapes))


def how_many_experiments(df, initial_cond='back', shapes=['I', 'T', 'H']):
    """
    plot how much data you already have
    :param df:, data frame with all the experiments
    :param initial_cond: which experiments to include
    :param shapes: which shapes to include
    """
    fig, ax = plt.subplots(1, len(shapes), sharey=True)

    for i, shape in enumerate(shapes):
        df_shape = df[df['shape'] == shape]
        ant_df = df_shape.loc[df_shape.groupby('solver').groups['ant']]
        ant_dict = ant_df.groupby('size').groups
        sizes_here = [s for s in sizes['ant'] if s in ant_dict.keys()]

        ax[i].bar(range(len(ant_dict)), [len(ant_dict[size]) for size in sizes_here], color='k')
        ax[i].set_xticks(range(len(ant_dict)))
        ax[i].set_xticklabels(sizes_here)
        ax[i].set_xlabel('maze size')
        ax[i].set_ylim(0, 120)
        ax[i].set_title(shape)

    ax[0].set_ylabel('number of experiments')
    save_fig(fig, 'how_many_experiments_' + initial_cond + '_' + ''.join(shapes))


if __name__ == '__main__':
    # untracked = {'humans': {'S': 14, 'L': 1}}
    how_many_experiments_SPT(df)
