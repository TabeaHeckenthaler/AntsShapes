from PhaseSpaces.PhaseSpace import PhaseSpace
from Directories import ps_path
from trajectory_inheritance import get

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

x = get('XL_SPT_dil2_sensing15', 'ps_simulation')
print(x.play.__doc__)  # this will print for you, how to use the .play method!
x.play(1, PhaseSpace=ps, ps_figure=fig, wait=200, interval=10)

k = 1