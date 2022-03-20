from Analysis.PathPy.Network import *
from DataFrame.choose_experiments import choose_experiments


if __name__ == '__main__':
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    fig, ax = plt.subplots()
    index = None

    for i, size in enumerate(exp_types[shape][solver]):
        paths = Paths(solver, size, shape, geometry)
        paths.load_paths(filenames=choose_experiments(solver, shape, size, geometry, communication=True).filenames)

        my_network = Network.init_from_paths(paths, solver, size, shape)
        my_network.get_results()
        my_network.plot_transition_matrix()

        if i == 0:
            t = my_network.t.sort_values(0, ascending=False)
            index = t.index
            ax.set_xticks(ticks=range(len(index)))
            ax.set_xticklabels(index)
        else:
            t = my_network.t.loc[index]

        t.plot(ax=ax, label=size)

    plt.show(block=False)
    ax.set_ylabel('humans: number of states to pass before solving')
    ax.legend(exp_types[shape][solver])
    fig.savefig(os.path.join(graph_dir(), 'human_expected_solving_time_no_self' + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')
