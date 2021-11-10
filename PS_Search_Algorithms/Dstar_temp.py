from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import get
from Directories import ps_path

# x = get('XL_SPT_dil0_sensing1')
x = get('XL_SPT_dil9_sensing4')

ps = PhaseSpace.PhaseSpace(x.solver, x.size, x.shape, name=x.size + '_' + x.shape)
ps.load_space(path=ps_path(x.size, x.shape))
ps.visualize_space()

x.play(ps=ps, step=3)
