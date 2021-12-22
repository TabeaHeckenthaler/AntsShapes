from trajectory_inheritance.trajectory import get
from Analysis.PathPy.network_functions import *
from pathpy.visualisation import plot, export_html
from os import path
from Directories import network_dir

if __name__ == '__main__':
    conf_space_labeled = load_labeled_conf_space()

    filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    trajectories = [get(filename) for filename in filenames]  # TODO: display trajectory in phase space... buggy?
    labels = [conf_space_labeled.reduces_labels(conf_space_labeled.label_trajectory(x)) for x in trajectories]
    paths = create_paths(labels)
    n = pp.Network.from_paths(paths)
    hon = create_higher_order_network(paths)

    export_html(n, path.join('network.html'))
    export_html(hon, path.join(network_dir, 'hon.html'))
