from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import exp_types

shape = 'SPT'
solver = 'ant'
sizes = ['Small Far']
new2021 = True

for size in exp_types[shape][solver]:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, new2021=new2021)
    conf_space.calculate_boundary(new2021=new2021)
    conf_space.save_space()
    # conf_space_labeled.visualize_states(reduction=10)
    # conf_space_labeled.visualize_transitions(reduction=10)

    DEBUG = 1

# TODO what is going on with smooth connector. See step by step in what direction he is moving.
#  Can I display the smooth connector?
#  Can I put the two together?
#  When I display the movies, does if help if I shift the center of mass?
#  If it does, you have to recalculate the Phase Space.

