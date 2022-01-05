from PhaseSpaces import PhaseSpace
from trajectory_inheritance.trajectory import get
from Analysis.States import States
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    conf_space = PhaseSpace.PhaseSpace('ant', 'XL', 'SPT', name='')
    conf_space.load_space()
    # conf_space.visualize_space(colormap='Oranges')

    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space)
    conf_space_labeled.load_space()
    # conf_space_labeled.save_labeled()

    x = get('XL_SPT_dil9_sensing4')
    labels = States(conf_space_labeled, x, step=x.fps)
    k = 1
