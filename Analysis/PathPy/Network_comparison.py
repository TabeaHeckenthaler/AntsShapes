from matplotlib import pyplot as plt
from trajectory_inheritance.exp_types import geometries, exp_types
from Analysis.PathPy.Paths import PathWithoutSelfLoops
from Analysis.PathPy.Network import Network
from Analysis.GeneralFunctions import graph_dir
import os


class Network_comparison:
    def __init__(self, networks: dict = None):
        self.networks = networks

    @classmethod
    def load_networks(cls) -> dict:
        networks = {solver: {} for solver in solvers}
        for solver in solvers:
            for size in sizes[solver]:
                paths = PathWithoutSelfLoops(solver, size, shape, geometries[solver])
                paths.load_paths()
                my_network = Network.init_from_paths(paths, solver, size, shape)
                my_network.get_results()
                networks[solver][size] = my_network
        return networks

    def plot_transition_matrices(self):
        for solver in self.networks.keys():
            fig, axs = plt.subplots(1, len(self.networks[solver]))
            for (size, network), ax in zip(self.networks[solver].items(), axs):
                network.plot_transition_matrix(title=size, axis=ax)

            directory = graph_dir() + os.path.sep + 'transition_matrix_' + solver + '.pdf'
            print('Saving transition matrix in ', directory)
            fig.savefig(directory)


shape = 'SPT'
solvers = ['human', 'ant']
sizes = exp_types[shape]

if __name__ == '__main__':
    my_networks = Network_comparison.load_networks()
    my_network_comparison = Network_comparison(my_networks)
    my_network_comparison.plot_transition_matrices()

    # TODO: Check whether the right solver number is involved (especially for human Medium).
    # TODO: Plot with equal state order
