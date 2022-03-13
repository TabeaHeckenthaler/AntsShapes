from ConfigSpace import ConfigSpace_Maze
from Analysis.PathPy.network_functions import *
from Analysis.States import States, states, forbidden_transition_attempts, allowed_transition_attempts
from trajectory_inheritance.exp_types import exp_types
import json


if __name__ == '__main__':
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
    sizes = exp_types[shape][solver]
    transition_matrices = {}

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))  # Create a figure with all the relevant data

    for size, axis in zip(sizes, axes.flatten()):
        if os.path.exists(size + '_transitions.txt'):
            with open(size + '_transitions.txt', 'r') as f:
                states_series = json.load(f)
                DEBUG = 1
        else:
            conf_space_labeled = ConfigSpace_Maze.PhaseSpace_Labeled(solver, size, shape, geometry)
            conf_space_labeled.load_eroded_labeled_space()
            # conf_space_labeled.visualize_states(reduction=5)
            conf_space_labeled.visualize_transitions(reduction=2)
            state_order = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)

            trajectories = get_trajectories(solver=solver, size=size, shape=shape, geometry=geometry)
            my_states = [States(conf_space_labeled, x, step=int(0.3 * x.fps)) for x in trajectories]
            states_series = [state.state_series for state in my_states]

            with open(size + '_transitions.txt', 'w') as json_file:
                json.dump(states_series, json_file)

            # my_network = Network(state_order, name='_'.join(['network', solver, size, shape]))
            # my_network.add_paths(paths)
            # transition_matrices[size] = my_network.transition_matrix().toarray()
            # # my_network.plot_network()
            # my_network.plot_transition_matrix(title=size, axis=axis)

            DEBUG = 1