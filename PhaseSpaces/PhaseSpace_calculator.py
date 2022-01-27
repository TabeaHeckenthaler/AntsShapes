from PhaseSpaces import PhaseSpace
from trajectory_inheritance.exp_types import exp_types

shape = 'SPT'
solver = 'ant'
sizes = ['XL']

for size in exp_types[shape][solver]:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, ('MazeDimensions_new2021_ant.xlsx',
                                                             'LoadDimensions_new2021_ant.xlsx'))
    conf_space.calculate_space()
    # conf_space.calculate_boundary()
    # conf_space.save_space()
    # conf_space_labeled.visualize_states(reduction=10)
    # conf_space_labeled.visualize_transitions(reduction=10)

    DEBUG = 1

# TODO what is going on with smooth connector. See step by step in what direction he is moving.
#  Can I display the smooth connector?
#  Can I put the two together?
#  When I display the movies, does if help if I shift the center of mass?
#  If it does, you have to recalculate the Phase Space.
#  submit the movies for image analysis.

