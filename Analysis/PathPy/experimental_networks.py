from PhaseSpaces import PhaseSpace
from Analysis.PathPy.network_functions import *


if __name__ == '__main__':
    solver = 'human'
    size = 'Large'
    shape = 'SPT'

    print(solver, size)
    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(solver, size, shape, new2021=True)
    conf_space_labeled.load_labeled_space()

    conf_space_labeled.visualize_states(reduction=4)
    conf_space_labeled.visualize_labels(reduction=4)
    print(conf_space_labeled.check_labels())


    # trajectories = get_trajectories(solver=solver, size=size, shape=shape, number=20)
    #
    # list_of_states = [States(conf_space_labeled, x, step=int(x.fps/2)) for x in trajectories]
    #
    # paths, n = pathpy_network(list_of_states)
    # T = Markovian_analysis(n)
    # t = absorbing_state_analysis(T)
    # series = pd.Series([t], ['t'], name=size)
    # results = results.append(series)
    # results.tojson(path.join(network_dir, 'Markov_results.json'))


# TODO: Look at returning to dead ends to define

