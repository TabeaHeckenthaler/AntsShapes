from ConfigSpace.ConfigSpace_Maze import *
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

factor = 1
#
x_range = (0, 29.35)
y_range = (0, 19.1)
pos_resolution = 0.1 / factor
theta_resolution = 0.013688856878386899 / factor
e_c_intersection = np.array([147, 60, 147]) * factor
ac_c_intersection = np.array([98, 75, 256]) * factor
#
# shape = np.array([294, 191, 459]) * factor
#
# # find high resolution
# mask = np.zeros(shape=shape).astype(bool)
# d = 10 * factor
# mask[e_c_intersection[0] * factor - d:e_c_intersection[0] * factor + d,
# e_c_intersection[1] * factor - d:e_c_intersection[1] * factor + d,
# e_c_intersection[2] * factor - d:e_c_intersection[2] * factor + d] = True

ps_bottleneck = ConfigSpace_Maze(solver='ant', size='XL', shape='SPT',
                                 geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                           'LoadDimensions_new2021_SPT_ant.xlsx'),
                                 x_range=x_range, y_range=y_range, pos_resolution=pos_resolution,
                                 theta_resolution=theta_resolution)
ps_bottleneck.calculate_space()
image = ps_bottleneck.space[100 * factor:-100 * factor, :, :459 // 2 * factor]
# er_space = ps_bottleneck.erode(image, radius=10)

distance = ndi.distance_transform_edt(image.astype(float))
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
distance_threshold = 0.08 * distance.max()
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
markers[distance > distance_threshold] = 0
labels = watershed(-distance, markers, mask=image)

# find 3 most common values in labels
unique, counts = np.unique(labels, return_counts=True)

# find 3 largest values in counts
n = 10
idx = np.argpartition(counts, -n)[-n:]
most_common_label = unique[idx]
most_common_counts = counts[idx]

fig = ps_bottleneck.new_fig()
from itertools import cycle
colormaps = cycle(['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples'])

for label, colormap in zip(most_common_label, colormaps):
    ps_bottleneck.visualize_space(space=labels == label, colormap=colormap, fig=fig)

import numpy as np
from skimage import morphology

skeleton = morphology.skeletonize(image)
narrowest_connection = np.count_nonzero(skeleton)
fig = ps_bottleneck.new_fig()
ps_bottleneck.visualize_space(space=image, colormap='Reds', fig=fig)
ps_bottleneck.visualize_space(space=skeleton, fig=fig)

DEBUG = 1