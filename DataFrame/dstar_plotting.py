from DataFrame.plot_dataframe import save_fig
from matplotlib import pyplot as plt
from Analysis.GeneralFunctions import three_D_plotting,non_duplicate_legend, colors
import matplotlib.pylab as pl


def difficulty(df, shapes, dil_radius=10, sensing_radius=5, measure='path length/exit size []'):
    from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
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
    save_fig(plt.gcf(), 'difficulty')


def dstar_figure(df, shape='SPT'):
    intersection = df.groupby(by='solver').groups['ps_simulation'].intersection(
        df.groupby(by='shape').groups[shape])

    df_dstar = df.iloc[intersection][['filename', 'path length/exit size []']]
    df_dstar[['sensing_radius']] = df_dstar.apply(lambda x: x.sensing_radius, axis=1)
    df_dstar[['dil_radius']] = df_dstar.apply(lambda x: x.dil_radius, axis=1)

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
    save_fig(fig, 'dstar_')
