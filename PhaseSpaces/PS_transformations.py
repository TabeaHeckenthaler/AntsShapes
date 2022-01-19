from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import get
from trajectory_inheritance.exp_types import exp_types
from Analysis.States import States
import numpy as np
from Directories import ps_path
from matplotlib import pyplot as plt
import os
from copy import copy
from Analysis.resolution import specific_resolution


def mask_around_tunnel(conf_space: PhaseSpace):
    factor = int(specific_resolution)
    mask = conf_space.empty_space()
    if conf_space.size == 'S':
        center = (factor * 220, factor * 70, factor * 150)
        radiusx, radiusy, radiusz = factor * 20, factor * 20, factor * 20

    else:
        center = (factor * 147, factor * 63, factor * 150)
        radiusx, radiusy, radiusz = factor * 9, factor * 12, factor * 10
    mask[center[0] - radiusx:center[0] + radiusx,
         center[1] - radiusy:center[1] + radiusy,
         center[2] - radiusz:center[2] + radiusz] = True
    return mask


if __name__ == '__main__':
    # only part of the shape
    solver, shape = 'ant', 'SPT'
    for size in ['XL', 'L', 'M', 'S']:
        conf_space_part = PhaseSpace.PhaseSpace(solver, size, shape, name='')

        # mask = mask_around_tunnel(conf_space_part)
        # conf_space_part.calculate_space(mask=mask)
        # new_space = copy(conf_space_part.space)
        #
        # conf_space_part.load_space()
        # # conf_space_part.visualize_space()
        # conf_space_part.visualize_space(space=new_space)
        # # conf_space_part.visualize_space(space=mask, colormap='Oranges')
        #
        conf_space_part.calculate_space()
        conf_space_part.calculate_boundary()
        conf_space_part.save_space()

    DEBUG = 1


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




