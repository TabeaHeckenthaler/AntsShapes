from Analysis.PathPy.Network import *
from DataFrame.choose_experiments import choose_experiments


def calculate_diffusion_time(i, index, filenames, paths, solver, size, shape, ax) -> tuple:
    paths.load_paths(filenames=filenames)

    my_network = Network.init_from_paths(paths, solver, size, shape)
    my_network.get_results()

    if i == 0:
        t = my_network.t.sort_values(0, ascending=False)
        index = t.index
        ax.set_xticks(ticks=range(len(index)))
        ax.set_xticklabels(index)
    else:
        t = my_network.t.reindex(index)
        t = t.loc[index]
    return t, index


def humans():
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    fig, ax = plt.subplots()
    index = None
    legend = []

    for i, size in enumerate(exp_types[shape][solver][:-1]):
        for communication in [True, False]:
            paths = Paths(solver, size, shape, geometry)
            filenames = choose_experiments(solver, shape, size, geometry, communication=communication).filename
            if len(filenames) > 0:
                t, index = calculate_diffusion_time(i, index, filenames, paths, solver, size, shape, ax)
                t.plot(ax=ax, label=size)
                legend.append(size + '_comm_' + str(communication))

    plt.show(block=False)
    ax.set_ylabel('humans: number of states to pass before solving')
    ax.legend(legend)
    fig.savefig(os.path.join(graph_dir(), 'human_expected_solving_time' + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')
    DEBUG = 1


def ants():
    solver, shape, geometry = 'ant', 'SPT', ('MazeDimensions_new2021_SPT_ant.xlsx',
                                             'LoadDimensions_new2021_SPT_ant.xlsx')

    fig, ax = plt.subplots()
    index = None
    legend = []

    for i, size in enumerate(exp_types[shape][solver][:-1]):
        paths = Paths(solver, size, shape, geometry)
        filenames = choose_experiments(solver, shape, size, geometry).filename
        t, index = calculate_diffusion_time(i, index, filenames, paths, solver, size, shape, ax)
        t.plot(ax=ax, label=size)
        legend.append(size)

    plt.show(block=False)
    ax.set_ylabel('ants: number of states to pass before solving')
    ax.legend(legend)
    fig.savefig(os.path.join(graph_dir(), 'ants_expected_solving_time' + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')
    DEBUG = 1


if __name__ == '__main__':
    ants()
