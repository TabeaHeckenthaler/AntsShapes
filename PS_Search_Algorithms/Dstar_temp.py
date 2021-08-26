from PhaseSpaces import PhaseSpace
from trajectory import Get

import os
import numpy as np

size = 'XL'
shape = 'SPT'
solver = 'ant'

data_dir = 'PhaseSpace\\' + solver

x = Get('Dlite_prior_knowledge_of_walls', 'dstar')
# x.play(1, 'Display', wait=20)

ps = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape)
ps.load_space(path=os.path.join(PhaseSpace.ps_dir, solver, ps.name + ".pkl"))
fig = ps.visualize_space(ps.name)
# x.play(1,
#        'Display',
#        PhaseSpace=ps,
#        ps_figure=fig)
fig = ps.draw_trajectory(fig, x.position, x.angle, scale_factor=0.2, color=(1, 0, 0))
fig = ps.draw_trajectory(fig, np.array([x.position[0]]), np.array([x.angle[0]]), scale_factor=1, color=(0, 0, 0))
fig = ps.draw_trajectory(fig, np.array([x.position[-1]]), np.array([x.angle[-1]]), scale_factor=1, color=(0, 0, 0))
print()
