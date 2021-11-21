import os
from Directories import PhaseSpaceDirectory
from PhaseSpaces.PhaseSpace import PhaseSpace
from trajectory_inheritance.trajectory import get

# x = get('XL_H_4100020_1_ants')
#
# name = x.size + '_' + x.shape
#
# point_particle = False
# if point_particle:
#     name = name + '_pp'
#
# path = os.path.join(PhaseSpaceDirectory, x.solver, name + ".pkl")
# ps = PhaseSpace(x.solver, x.size, x.shape, name=name)
# ps.load_space(path=path)
# ps.visualize_space()
#
# x.play(ps=ps, step=10)


# x = get('XL_H_4100020_1_ants')

shape = 'H'
name = "XL" + '_' + shape

point_particle = False
if point_particle:
    name = name + '_pp'

ps = PhaseSpace('ant', 'XL', shape, name=name)
ps.load_space()
ps.visualize_space()
k = 1
