import numpy as np
from scipy import ndimage
import copy


def dilation(ps, radius=8):
    ps_copy = copy.deepcopy(ps)
    struct = np.ones([radius for _ in range(ps.space.ndim)], dtype=bool)
    ps_copy.space = np.array(~ndimage.binary_dilation(~np.array(ps_copy.space, dtype=bool), structure=struct), dtype=int)

    # fig = ps_copy.visualize_space(ps_copy.name)
    return ps_copy
