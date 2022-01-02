import pathpy as pp

p = pp.Paths()
p.add_path(('a', 'c', 'd', 'c', 'd'), frequency=1)
p.add_path(('a', 'c', 'd', 'c'), frequency=1)


n = pp.Network.from_paths(p)

pp.visualisation.html.plot_diffusion(p)
