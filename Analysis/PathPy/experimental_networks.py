from ConfigSpace import ConfigSpace_Maze
from Analysis.PathPy.network_functions import *
from Analysis.States import States, states, forbidden_transition_attempts, allowed_transition_attempts
from trajectory_inheritance.exp_types import exp_types


if __name__ == '__main__':
    solver, shape = 'human', 'SPT'
    geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    for size in exp_types[shape][solver]:
        conf_space_labeled = ConfigSpace_Maze.PhaseSpace_Labeled(solver, size, shape, geometry)
        conf_space_labeled.load_labeled_space()
        conf_space_labeled.visualize_states()
        state_order = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)

        trajectories = get_trajectories(solver=solver, size=size, shape=shape, geometry=geometry)
        list_of_states = [States(conf_space_labeled, x, step=int(x.fps)) for x in trajectories]
        paths = [states.combine_transitions(states.state_series) for states in list_of_states]

        my_network = Network(state_order, name='_'.join(['network', solver, size, shape]))
        my_network.add_paths(paths)
        my_network.plot_network()
        my_network.plot_transition_matrix()

