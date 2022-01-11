from PhaseSpaces import PhaseSpace
from Analysis.PathPy.network_functions import *
from Analysis.States import States
import pandas as pd

if __name__ == '__main__':
    solver, size, shape = 'human', 'Large', 'SPT'

    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(solver, size, shape, new2021=True)
    conf_space_labeled.load_labeled_space()

    # conf_space_labeled.visualize_states()
    # conf_space_labeled.visualize_transitions(reduction=5)
    # print(conf_space_labeled.check_labels())

    trajectories = get_trajectories(solver=solver, size=size, shape=shape)
    list_of_states = [States(conf_space_labeled, x, step=int(2 * x.fps)) for x in trajectories]

    # analysis of state_series
    paths_state, n_state = pathpy_network([states.combine_transitions(states.state_series) for states in list_of_states])
    T_state = Markovian_analysis(n_state)
    t_state = absorbing_state_analysis(T_state)

    # analysis of time_series
    paths_time, n_time = pathpy_network([states.time_series for states in list_of_states])
    T_time = Markovian_analysis(n_time)
    t_time = absorbing_state_analysis(T_time) * list_of_states[-1].time_step

    DEBUG = 1


# TODO: Look at returning to dead ends to define

