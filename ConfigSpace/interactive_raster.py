from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ConfigSpace import Raster_ConfigSpace_Maze
from Setup.Maze import Maze

shape = 'SPT'
solver = 'human'
geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
size = 'Large'

maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)
raster = Raster_ConfigSpace_Maze.Raster_ConfigSpace_Maze(maze)

xbounds = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
ybounds = (0, maze.arena_height)


def get(theta):
    return raster(theta, 415, 252, xbounds, ybounds).T  # transpose since matrix indexing is depth-width and not xy


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 15

plt.style.use('dark_background')

fig, ax = plt.subplots()

im = ax.imshow(get(0), extent=xbounds+ybounds, aspect='auto', interpolation='nearest', cmap='gray', origin='lower')

plt.subplots_adjust(bottom=0.25)
axslider = plt.axes((0.1, 0.1, 0.8, 0.03))
slider = Slider(
    ax=axslider,
    label=r'$\theta$',
    valmin=0,
    valmax=2*pi,
    valinit=0,
)
slider.on_changed(lambda theta: im.set_data(get(theta)))

plt.show()
