from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import get, exp_types
from Analysis.States import States
import numpy as np
from Directories import ps_path
from matplotlib import pyplot as plt
import os


def mask_around_tunnel(conf_space):
    mask = np.zeros((293, 192, 454), dtype=bool)
    center = (147, 63, 150)
    radiusx, radiusy, radiusz = 10, 15, 15
    mask[center[0] - radiusx:center[0] + radiusx,
         center[1] - radiusy:center[1] + radiusy,
         center[2] - radiusz:center[2] + radiusz] = True
    return mask


if __name__ == '__main__':
    # only part of the shape
    solver, size, shape = 'ant', 'XL', 'SPT'
    conf_space_part = PhaseSpace.PhaseSpace(solver, size, shape, name='')
    conf_space_part.load_space()
    mask = mask_around_tunnel(conf_space_part)
    conf_space_part.calculate_boundary(mask=mask)

    conf_space_part.visualize_space()
    conf_space_part.visualize_space(space=mask, colormap='Oranges')



# for shape, solvers in exp_types.items():
#     for solver, sizes in solvers.items():
#         for size in sizes:
#             conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='')
#             conf_space.load_space()
#             conf_space.visualize_space(colormap='Greys')

# conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space)
# conf_space_labeled.load_space()
# # conf_space_labeled.save_labeled()
#
# x = get('XL_SPT_dil9_sensing4')
# labels = States(conf_space_labeled, x, step=x.fps)
# k = 1




