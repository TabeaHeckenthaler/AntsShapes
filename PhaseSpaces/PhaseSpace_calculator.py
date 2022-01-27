from PhaseSpaces import PhaseSpace
from trajectory_inheritance.exp_types import exp_types

shape = 'I'
solver = 'ant'
sizes = ['XL']

for size in exp_types[shape][solver]:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, ('MazeDimensions_ant.xlsx', 'LoadDimensions_ant.xlsx'))
    conf_space.load_space()
    conf_space.visualize_space()
    # conf_space.calculate_space()
    # conf_space.calculate_boundary()
    # conf_space.save_space()
    # conf_space_labeled.visualize_states(reduction=10)
    # conf_space_labeled.visualize_transitions(reduction=10)

    DEBUG = 1

