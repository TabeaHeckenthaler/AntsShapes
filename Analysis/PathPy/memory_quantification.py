from Analysis.PathPy.Network import *


def plot_diffusion_speed_up(speed_up_dict):
    plt.plot(speed_up_dict)
    plt.show(block=False)
    ax.set_ylabel('humans: number of states to pass before solving')
    ax.legend(exp_types[shape][solver])
    fig.savefig(os.path.join(graph_dir(), 'human_expected_solving_time_no_self' + '.png'),
                format='png', pad_inches=0.5, bbox_inches='tight')

shape = 'SPT'
geometries = {
    ('ant', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')): ['XL', 'L', 'M', 'S'],
    ('human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')): ['Large', 'Medium', 'Small Far'],
}

fig, ax = plt.subplots()

index = None
networks = []
speed_up_dict = {'ant': {}, 'human': {}}

for (solver, geometry), sizes in list(geometries.items()):
    print(solver)
    for size in sizes:
        paths = PathWithoutSelfLoops(solver, size, shape, geometry)
        paths.load_paths()
        my_network = Network.init_from_paths(paths, solver, shape, size)
        my_network.markovian_analysis()
        speed_up_dict[solver][size] = my_network.diffusion_speed_up()
        print(size)
        print(speed_up_dict[solver][size])

plot_diffusion_speed_up(networks)
DEBUG = 1
