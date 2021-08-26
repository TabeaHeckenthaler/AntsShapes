import os
from PhaseSpaces.PhaseSpace import PhaseSpace, ps_dir


shape = 'SPT'
size = 'XL'
point_particle = False
solver = 'ant'

name = size + '_' + shape

if point_particle:
    name = name + '_pp'

path = os.path.join(ps_dir, solver, name + ".pkl")
ps = PhaseSpace(solver, size, shape, name=name)
ps.load_space(path=os.path.join(path))
ps.visualize_space(ps.name)

print(ps.name)

#  e7e1dbf266e6d5d96177c95b825eefdf40e7fd22