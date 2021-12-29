from Analysis.PathPy.network_functions import *
from pathpy.visualisation import plot, export_html
from os import path
from Directories import network_dir

solver = 'human'
size = 'Large'
shape = 'SPT'

if __name__ == '__main__':
    # load labeled configuration space
    conf_space_labeled = load_labeled_conf_space(solver=solver, size=size, shape=shape)

    # load relevant experimental paths
    trajectories = get_trajectories(solver=solver, size=size, shape=shape)

    # label the trajectory according to conf_space_labeled
    labels = [conf_space_labeled.reduces_labels(conf_space_labeled.label_trajectory(x)) for x in trajectories]

    # create paths and network
    paths = create_paths(labels)
    n = pp.Network.from_paths(paths)
    hon = create_higher_order_network(paths)

    # display network
    export_html(n, path.join('network.html'))
    export_html(hon, path.join(network_dir, 'hon.html'))
