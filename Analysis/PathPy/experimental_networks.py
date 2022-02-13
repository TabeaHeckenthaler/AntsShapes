from ConfigSpace import ConfigSpace_Maze
from Analysis.PathPy.network_functions import *
from Analysis.States import States, states, forbidden_transition_attempts, allowed_transition_attempts
from trajectory_inheritance.exp_types import exp_types
import pandas as pd


if __name__ == '__main__':
    solver, shape = 'ant', 'SPT'
    geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')
    for size in exp_types[shape][solver]:
        trajectories = get_trajectories(solver=solver, size=size, shape=shape, geometry=geometry)

        conf_space_labeled = ConfigSpace_Maze.PhaseSpace_Labeled(solver, size, shape, geometry)
        conf_space_labeled.load_labeled_space()

        # conf_space_labeled.visualize_states()
        # conf_space_labeled.visualize_transitions()
        # print(conf_space_labeled.check_labels())

        list_of_states = [States(conf_space_labeled, x, step=int(x.fps)) for x in trajectories]

        # analysis of state_series
        paths_state, network_state = pathpy_network([states.combine_transitions(states.state_series)
                                                     for states in list_of_states])
        plot_network(network_state, name='_'.join(['network', solver, size, shape]))

        T_state = Markovian_analysis(network_state)
        # state_order = list(network_state.node_to_name_map().keys()) + ['i']
        state_order = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)
        plot_transition_matrix(T_state, state_order, name='_'.join(['T', solver, size, shape]))
        # t_state = absorbing_state_analysis(T_state)

        # analysis of time_series
        # paths_time, n_time = pathpy_network([states.cut_at_end(states.time_series) for states in list_of_states])
        # T_time = Markovian_analysis(n_time)
        # t_time = absorbing_state_analysis(T_time) * list_of_states[-1].time_step
        #
        # DEBUG = 1


