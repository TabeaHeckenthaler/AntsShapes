from PhaseSpaces import PhaseSpace, PS_transformations
from trajectory_inheritance.trajectory import get


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

ps_dil = PS_transformations.dilation(ps, radius=20)
ps_dil.visualize_space(fig, colormap='Oranges')
k = 1
# x.play(ps=ps, step=3)
