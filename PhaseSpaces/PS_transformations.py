from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import get, exp_types
from Analysis.States import States
import numpy as np
from Directories import ps_path
from matplotlib import pyplot as plt
import os


if __name__ == '__main__':
    for shape, solvers in exp_types.items():
        for solver, sizes in solvers.items():
            for size in sizes:
                conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='')

    # conf_space.visualize_space(colormap='Oranges')

    # conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space)
    # conf_space_labeled.load_space()
    # # conf_space_labeled.save_labeled()
    #
    # x = get('XL_SPT_dil9_sensing4')
    # labels = States(conf_space_labeled, x, step=x.fps)
    # k = 1
