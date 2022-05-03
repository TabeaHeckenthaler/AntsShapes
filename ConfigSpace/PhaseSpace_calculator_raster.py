import numpy as np

from ConfigSpace import Raster_ConfigSpace_Maze
from Setup.Maze import Maze

shape = 'SPT'
solver = 'human'
geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
size = 'Large'

maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)

raster = Raster_ConfigSpace_Maze.Raster_ConfigSpace_Maze(maze)

# use same shape and bounds as original phase space calculator
space_shape = (415, 252, 616)  # x, y, theta.

xbounds = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
ybounds = (0, maze.arena_height)


final_arr = np.empty(space_shape, bool)  # better to use space_shape[::-1] in terms of access speed
thet_arr = np.linspace(0, 2*np.pi, space_shape[2], False)

# make the final array slice by slice
for i, theta in enumerate(thet_arr):
    if not i % 50: print(f"{i}/{space_shape[2]}")
    final_arr[:, :, i] = raster(theta, space_shape[0], space_shape[1], xbounds, ybounds)


# then plot it

import pyvista as pv
from skimage import measure  # for marching cubes

pv.set_plot_theme('dark')
plotter = pv.Plotter()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.enable_anti_aliasing()

#plotter.add_volume(final_arr.astype(float)*100000, clim=[0, 1], show_scalar_bar=False, opacity='linear', resolution=space_shape)

verts, faces, normals, values = measure.marching_cubes(final_arr, .5)
verts /= space_shape
verts[:, 0] *= xbounds[1] - xbounds[0]; verts[:, 0] += xbounds[0]
verts[:, 1] *= ybounds[1] - ybounds[0]; verts[:, 0] += ybounds[0]
verts[:, 2] *= 2*np.pi*maze.average_radius()
faces = (np.concatenate((np.ones(faces.shape[:1] + (1,), int)*3, faces), 1)).flatten()

plotter.add_mesh(pv.PolyData(verts, faces), color='white', lighting=True, smooth_shading=True)

plotter.show_bounds(grid='back', location='front',
                    xlabel='x', ylabel='y', zlabel='z',
                    all_edges=True)

plotter.show()


