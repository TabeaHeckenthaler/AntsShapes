from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
import numpy as np
from mayavi import mlab

radius = 3
ps = ConfigSpace_SelectedStates(solver='ant', size='XL', shape='SPT', geometry=(
                          'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
ps.load_final_labeled_space()
#
# # Find the smallest bottleneck area...
# from scipy.ndimage.measurements import label
# bottle = np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'e')
# eroded_bottle = ps.erode(space=bottle, radius=3)
# structure = np.ones((3, 3, 3), dtype=int)
# labeled, ncomponents = label(eroded_bottle, structure)
#
# space1 = labeled == 1
# space2 = labeled == 2
# space3 = labeled == 3
# DEBUG = 1

# Find the intersection of ac and c
ac = ps.space_labeled == 'ac'
c = ps.space_labeled == 'c'
ac_c_intersection = np.logical_and(ps.dilate(space=ac, radius=radius), ps.dilate(space=c, radius=radius))

# Find the intersection of c and e
e = ps.space_labeled == 'e'
e_c_intersection = np.logical_and(ps.dilate(space=e, radius=radius), ps.dilate(space=c, radius=radius))

# Find the rest of the area
boundary_and_forbidden = ps.dilate(space=~ps.space, radius=radius)
boundary = np.logical_and(boundary_and_forbidden, ps.space)
cg = ps.space_labeled == 'cg'
rest = np.logical_and(~ps.space, np.logical_or(ps.dilate(space=c, radius=radius), ps.dilate(space=cg, radius=radius)))

fig = ps.new_fig()
ps.visualize_space(space=rest, fig=fig)
ps.visualize_space(space=e, fig=fig, colormap='Reds')
ps.visualize_space(space=c, fig=fig, colormap='Greens')
mlab.show()

fig = ps.new_fig()
ps.visualize_space(space=rest, fig=fig)
ps.visualize_space(space=ac_c_intersection, fig=fig, colormap='Reds')
ps.visualize_space(space=e_c_intersection, fig=fig, colormap='Greens')
mlab.show()

# Find the area of the intersection
ac_area = np.sum(ac_c_intersection)
e_area = np.sum(e_c_intersection)
rest_area = np.sum(rest)
print('ac_area:', ac_area)
print('e_area:', e_area)
print('rest_area:', rest_area)

# Find the circumference
ac_circumference = np.logical_and(boundary, ac_c_intersection)
e_circumference = np.logical_and(boundary, e_c_intersection)

fig = ps.new_fig()
ps.visualize_space(space=ac_circumference, fig=fig, colormap='Reds')
ps.visualize_space(space=e_circumference, fig=fig, colormap='Greens')
mlab.show()

ac_circ = np.sum(ac_circumference)
e_circ = np.sum(e_circumference)
print('ac_circ:', ac_circ)
print('e_circ:', e_circ)
DEBUG = 1
