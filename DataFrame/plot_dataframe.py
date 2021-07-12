import pandas as pd
from DataFrame.create_dataframe import df_dir
from matplotlib import pyplot as plt
from Analysis_Functions.GeneralFunctions import graph_dir, non_duplicate_legend, three_D_plotting, colors
from os import path
import numpy as np
from matplotlib.transforms import Affine2D
from trajectory import solvers, Get
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_json(df_dir + '.json')
df_gr_solver = df.groupby(by='solver')
df_gr_shape = df.groupby(by='shape')

"""
Path length and Number of attempts (H, I, T)
"""


def ant_HIT_figure_path_length():
    sizes_order = ['XS', 'S', 'M', 'L', 'SL', 'XL']
    df_ant = df.loc[df_gr_solver.groups['ant'], ['maze size',
                                                 'shape',
                                                 'path_length_during_attempts [length unit]',
                                                 'exit size [length unit]',
                                                 'Attempts']]

    df_ant_HIT = df_ant.loc[df_ant['shape'].isin(['H', 'I', 'T'])].copy()

    df_ant_HIT['path length norm by exit size during attempt []'] = df_ant_HIT.apply(
        lambda x: x['path_length_during_attempts [length unit]'] / x['exit size [length unit]'], axis=1)
    df_ant_HIT['Attempt Number'] = df_ant_HIT.apply(
        lambda x: len(x['Attempts']), axis=1)

    y_axis = ['path length norm by exit size during attempt []', 'Attempt Number']
    for y in y_axis:
        group = df_ant_HIT.groupby(['maze size', 'shape'])[[y]]
        means = group.mean().unstack().reindex(sizes_order)
        sem = group.sem().unstack().reindex(sizes_order)

        fig, ax = plt.subplots()
        for shape, shift in zip(means.columns.get_level_values('shape'), [-0.1, 0.0, 0.1]):
            trans = Affine2D().translate(shift, 0.0) + ax.transData
            ax.errorbar(sizes_order,
                        np.array(means[y][shape]),
                        yerr=np.array(sem[y][shape]),
                        linestyle='',
                        marker='o',
                        transform=trans)

        legend = ['shape: ' + str(bo) for bo in means.columns.get_level_values('shape').values]
        ax.legend(legend)
        ax.set_ylabel(means.columns[0][0])
        fig.savefig(graph_dir() + path.sep + 'ants_' + y + '.pdf',
                    format='pdf', pad_inches=1, bbox_inches='tight')
        fig.savefig(graph_dir() + path.sep + 'ants_' + y + '.svg',
                    format='svg', pad_inches=1, bbox_inches='tight')


"""
Ant Maze: Path length SPT divided by winner and loser
"""


def Carrier_Number_Binning(df, solver, number_of_bins=5):
    bin_name = 'bin_name'
    if solver == 'human' or solver == 'ant':
        sorter_dict = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
        df[bin_name] = df['maze size'].map(sorter_dict)

        df.loc[(df['average Carrier Number'] < 3) & (df['maze size'] == 'M'), bin_name] = 0.5

        return df.sort_values(by=bin_name).reset_index(drop=True).copy()
    else:
        bin_content = int(np.ceil(len(df) / number_of_bins))
        sorted_df = df.sort_values(by='average Carrier Number').reset_index(drop=True).copy()
        sorted_df[bin_name] = [ii for ii in range(number_of_bins) for _ in range(bin_content)][:len(df)]

        # check bin boundaries
        def set_boundary_group_indices(sorted_df, i):
            aCN = int(sorted_df[['average Carrier Number']].iloc[i])
            indices = sorted_df.groupby(by='average Carrier Number').get_group(aCN).index
            in_group = sorted_df[bin_name].iloc[indices[0]]

            sorted_df.loc[indices, bin_name] = in_group
            return sorted_df

        for i in range(bin_content, len(df), bin_content):
            sorted_df = set_boundary_group_indices(sorted_df, i)
        return sorted_df


def SPT_figure(measure='path length/exit size []'):
    fig, ax = plt.subplots()

    for solver in solvers:
        df_solver_SPT = df.loc[(df['solver'] == solver) & (df['shape'] == 'SPT') & (df['winner']),
                               ['filename', 'maze size', measure, 'average Carrier Number',
                                'solver', 'communication']].copy()
        sorted_df = Carrier_Number_Binning(df_solver_SPT, solver)
        group = sorted_df.groupby(['bin_name', 'communication'])

        means = group.mean()
        sem = group.sem()

        settings = {'linestyle': '', 'marker': 'o', 'c': colors[solver]}
        if solver != 'human':
            ax.errorbar(means['average Carrier Number'],
                        means[measure],
                        xerr=np.array(sem['average Carrier Number']),
                        yerr=np.array(sem[measure]),
                        **settings, label=solver
                        )
        if solver == 'human':
            without_special_mean = means[~(means.index.get_level_values(0) == 0.5)]
            without_special_sem = sem[~(means.index.get_level_values(0) == 0.5)]
            comm_index = without_special_mean.index.get_level_values('communication')
            ax.errorbar(without_special_mean.loc[comm_index]['average Carrier Number'],
                        without_special_mean.loc[comm_index][measure],
                        xerr=np.array(without_special_sem.loc[comm_index]['average Carrier Number']),
                        yerr=np.array(without_special_sem.loc[comm_index][measure]),
                        **settings, label=solver + ' communication', mfc='w',
                        )
            ax.errorbar(without_special_mean.loc[comm_index == False]['average Carrier Number'],
                        without_special_mean.loc[comm_index == False][measure],
                        xerr=np.array(without_special_sem.loc[comm_index == False]['average Carrier Number']),
                        yerr=np.array(without_special_sem.loc[comm_index == False][measure]),
                        **settings, label=solver + ' no communication',
                        )

            # Special: M with only 1-2 participants
            special_mean = means[means.index.get_level_values(0) == 0.5]
            special_sem = sem[means.index.get_level_values(0) == 0.5]
            comm_index = special_mean.index.get_level_values('communication')
            ax.errorbar(special_mean.loc[comm_index]['average Carrier Number'],
                        special_mean.loc[comm_index][measure],
                        xerr=np.array(special_sem.loc[comm_index]['average Carrier Number']),
                        yerr=np.array(special_sem.loc[comm_index][measure]),
                        linestyle='', marker='x', c=colors[solver], label=solver + ' communication M', mfc='w',
                        )
            ax.errorbar(special_mean.loc[comm_index == False]['average Carrier Number'],
                        special_mean.loc[comm_index == False][measure],
                        xerr=np.array(special_sem.loc[comm_index == False]['average Carrier Number']),
                        yerr=np.array(special_sem.loc[comm_index == False][measure]),
                        linestyle='', marker='x', c=colors[solver], label=solver + ' no communication M',
                        )

    ax.legend()
    ax.set_xlabel('average Carrier Number')
    ax.set_ylabel(measure)

    fig.savefig(graph_dir() + path.sep + 'SPT_different_solver.pdf',
                format='pdf', pad_inches=1, bbox_inches='tight', )
    fig.savefig(graph_dir() + path.sep + 'SPT_different_solver.svg',
                format='svg', pad_inches=1, bbox_inches='tight')


"""
Human Maze: Path length SPT divided by communication and non-communication
"""


def human_figure():
    sizes_order = ['S', 'M', 'L']
    # df_human = df.iloc[df_gr_solver.groups['human']][['maze size', 'communication', 'path length/exit size []']]
    df_human = df.groupby('solver').getgroup('human')[['maze size', 'communication', 'path length/exit size []']]
    group = df_human.groupby(by=['maze size', 'communication'])
    means = group.mean().unstack().reindex(sizes_order)
    sem = group.sem().unstack().reindex(sizes_order)

    fig, ax = plt.subplots()
    means.plot(kind='bar', rot=0, ax=ax, yerr=sem)

    legend = ['comm: ' + str(bo) for bo in means.columns.get_level_values('communication').values]
    ax.legend(legend)
    ax.set_ylabel(means.columns[0][0])
    fig.savefig(graph_dir() + path.sep + 'humans_path_length' + '.pdf',
                format='pdf', pad_inches=1, bbox_inches='tight')
    fig.savefig(graph_dir() + path.sep + 'humans_path_length' + '.svg',
                format='svg', pad_inches=1, bbox_inches='tight')


"""
Human Maze: Path length SPT divided by communication and non-communication
"""


def dstar_figure(shape='SPT'):
    from Classes_Experiment.mr_dstar import sensing_radius, dil_radius
    intersection = df.groupby(by='solver').groups['dstar'].intersection(
        df.groupby(by='shape').groups[shape])

    df_dstar = df.iloc[intersection][['filename', 'path length/exit size []']]
    df_dstar[['sensing_radius']] = df_dstar.apply(lambda x: sensing_radius(x['filename']), axis=1)
    df_dstar[['dil_radius']] = df_dstar.apply(lambda x: dil_radius(x['filename']), axis=1)

    group = df_dstar.groupby(['sensing_radius', 'dil_radius'])
    means = group.mean().unstack()

    fig, ax = plt.subplots()
    colors = pl.cm.jet(np.linspace(0, 1, len(means)))
    means.plot(ax=ax,
               # logy=True,
               color=colors)

    legend = ['dilation: ' + str(bo) for bo in means.columns.get_level_values('dil_radius').values]
    ax.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel(means.columns[0][0])
    fig.savefig(graph_dir() + path.sep + 'dstar_' + shape + '.pdf',
                format='pdf', pad_inches=1, bbox_inches='tight')
    fig.savefig(graph_dir() + path.sep + 'dstar_' + shape + '.svg',
                format='svg', pad_inches=1, bbox_inches='tight')


for shape in ['I', 'T', 'H', 'SPT']:
    dstar_figure(shape=shape)


def difficulty(df, shapes, dil_radius=10, sensing_radius=5, measure='path length/exit size []'):
    from Classes_Experiment.mr_dstar import filename_dstar
    """ open figure and set colours """
    ax = plt.axes(projection='3d')

    """ reduce dataframe """
    df = df[['filename', 'shape', 'size', 'solver', 'average Carrier Number', measure]].copy()

    """ make the x axis """
    x_axis_dict = {}
    for shape in shapes:
        dstar_solution = filename_dstar('XL', shape, dil_radius, sensing_radius)
        shape_x = df.loc[df['filename'] == dstar_solution][[measure]].values[0][0]
        x_axis_dict[shape] = shape_x
        # plt.axvline(x=shape_x)
        ax.text(x=shape_x, y=10, z=0,
                s=shape,
                horizontalalignment='center',
                color='black',
                )

    """ calculate values """
    means = df.groupby(by=['shape', 'solver', 'size', ]).mean()
    sem = df.groupby(by=['shape', 'solver', 'size', ]).sem()

    # sizes = ['XL', 'Large', '']
    # for parameter, level in zip([shapes, sizes], [0, 2]):
    for parameter, level in zip([shapes], [0]):
        to_drop = [par for par in means.index.get_level_values(level).drop_duplicates() if par not in parameter]
        means = means.drop(level=level, index=to_drop)
        sem = sem.drop(level=level, index=to_drop)

    """ plot them """
    plt.show(block=False)
    means['x_axis'] = means.index.get_level_values('shape').map(x_axis_dict)

    for solver, indices in means.groupby(by='solver').indices.items():
        three_D_plotting(means.iloc[indices]['x_axis'].values,
                         means.iloc[indices][measure].values,
                         means.iloc[indices]['average Carrier Number'].values,
                         # np.zeros(means.iloc[indices][measure].values.shape),
                         yerr=sem.iloc[indices][measure].values,
                         color=colors[solver],
                         label=solver,
                         ax=ax)

        # old
        # ax.errorbar(x=x, y=y, z=z, yerr=yerr,
        #             linestyle='',
        #             marker='*',
        #             ecolor=colors[solver],
        #             c=colors[solver],
        #             label=solver
        #             )

    """ legend etc., save figure """
    non_duplicate_legend(ax)
    ax.set_xlabel('difficulty')
    ax.set_ylabel('path length/exit size, []')
    ax.set_zlabel('group size')
    # ax.set_yscale('log')

    plt.savefig(graph_dir() + path.sep + 'difficulty.pdf',
                format='pdf', pad_inches=1, bbox_inches='tight')
    plt.savefig(graph_dir() + path.sep + 'difficulty.svg',
                format='svg', pad_inches=1, bbox_inches='tight')


# difficulty(df, ['H', 'I', 'T'], dil_radius=0, sensing_radius=5)
difficulty(df, ['SPT', 'H', 'I', 'T'], dil_radius=0, sensing_radius=5)
# SPT_figure()

# ant_HIT_figure_path_length()