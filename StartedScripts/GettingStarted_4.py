import os
from PhaseSpaces.PhaseSpace import PhaseSpace
from Directories import ps_path
from trajectory import Get

shape = 'SPT'
size = 'XL'
point_particle = False
solver = 'ant'

name = size + '_' + shape

if point_particle:
    name = name + '_pp'

ps = PhaseSpace(solver, size, shape, name=name)
ps.load_space(path=ps_path(size, shape, solver))
fig = ps.visualize_space(ps.name)

print(ps.name)

x = Get('XL_SPT_dil2_sensing15', 'dstar', )
x.play(1, PhaseSpace=ps, ps_figure=fig, wait=20, interval=10)

k = 1