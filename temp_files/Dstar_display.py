from PhaseSpaces import PhaseSpace, PS_transformations
from trajectory_inheritance.trajectory import get
from copy import copy


x = get('XL_SPT_dil0_sensing1')
# x = get('XL_SPT_dil9_sensing4')

# ps_pp = PhaseSpace.PhaseSpace(x.solver, x.size, x.shape, name=x.size + '_' + x.shape + '_pp')
# ps_pp.load_space(point_particle=True)
# fig = ps_pp.new_fig()
# ps_pp.visualize_space(fig=fig, colormap='Blues')

ps = PhaseSpace.PhaseSpace(x.solver, x.size, x.shape, name=x.size + '_' + x.shape)
ps.load_space(point_particle=False)
fig = ps.new_fig()
ps.visualize_space(fig, colormap='Greys')

ps_dil = ps.dilate(ps.space, radius=20)
ps.visualize_space(fig, colormap='Oranges', space=ps_dil)

