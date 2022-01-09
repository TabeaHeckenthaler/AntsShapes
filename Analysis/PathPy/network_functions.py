from PhaseSpaces import PhaseSpace
import pathpy as pp
from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get
from matplotlib import pyplot as plt
import igraph
from pathpy.visualisation import export_html
from Directories import network_dir
from os import path


def load_labeled_conf_space(solver: str = 'ant', size: str = 'XL', shape: str = 'SPT', new2021: bool = True) \
        -> PhaseSpace:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='', new2021=new2021)
    conf_space.load_space(new2021=new2021)
    # conf_space.visualize_space()

    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space)
    conf_space_labeled.load_labeled_space()
    conf_space_labeled.visualize_states(reduction=10)
    print(conf_space_labeled.check_labels())
    return conf_space_labeled


def create_paths(labels) -> pp.paths:
    paths = pp.Paths()
    [paths.add_path(label) for label in labels]
    return paths


def create_higher_order_network(paths, k=2) -> pp.Network:
    hon = pp.HigherOrderNetwork(paths, k=k, null_model=True)

    # for e in hon.edges:
    #     print(e, hon.edges[e])
    return hon


def get_trajectories(solver='human', size='Large', shape='SPT', number: int = 10) -> list:
    """
    Get trajectories based on the trajectories saved in myDataFrame.
    :param number: How many trajectories do you want to have in your list?
    :param solver: str
    :param size: str
    :param shape: str
    :return: list of objects, that are of the class or subclass Trajectory
    """
    df = myDataFrame[
        (myDataFrame['size'] == size) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver)]

    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]


def plot_transition_matrix(T, state_order):
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(T)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(state_order)
    ax.set_yticklabels(state_order)


def Network2igraph(network):
    """
    Returns an igraph Graph object which represents
    the k-th layer of a multi-order graphical model.
    """
    g = igraph.Graph(directed=True)

    for e in network.edges:
        if g.vcount() == 0 or e[0] not in g.vs()["name"]:
            g.add_vertex(e[0])
        if g.vcount() == 0 or e[1] not in g.vs()["name"]:
            g.add_vertex(e[1])
        g.add_edge(e[0], e[1], weight=network.edges[e]['weight'])
    return g


def plot_network(n: pp.Network, name: str = 'network') -> None:
    """
    Save .html of the network.
    :param n: Network
    :param name: name, under which the .html will be saved
    """
    g1 = Network2igraph(n)
    visual_style = {"layout": g1.layout_auto(), "vertex_label": g1.vs["name"], "edge_label": g1.es["weight"],
                    "width": 3200, "height":3200,
                    # "d3js_path": # TODO
                    # "edge_width": 1
                    "label_color": "#010101",
                    # "edge_color": "#010101"
                    }
    export_html(n, path.join(network_dir, name + '.html'), **visual_style)
