from matplotlib import pyplot as plt
from trajectory_inheritance.exp_types import geometries, exp_types
from Analysis.PathPy.Paths import PathWithoutSelfLoops
from Analysis.PathPy.Network import Network
from Analysis.GeneralFunctions import graph_dir
from Analysis.PathPy.SPT_states import states, forbidden_transition_attempts, allowed_transition_attempts
import os
import numpy as np


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
                # we could also do: get results, if we are sure that we saved the right thing
                my_network.markovian_analysis()
                networks[solver][size] = my_network
        return networks

    def plot_transition_matrices(self):
        state_order = states[1:] + allowed_transition_attempts + forbidden_transition_attempts
        for solver in self.networks.keys():
            fig, axs = plt.subplots(1, len(self.networks[solver]))
            for (size, network), ax in zip(self.networks[solver].items(), axs):
                network.plot_transition_matrix(title=size, axis=ax, state_order=state_order)

            directory = graph_dir() + os.path.sep + 'transition_matrix_' + solver + '.pdf'
            print('Saving transition matrix in ', directory)
            fig.savefig(directory)

    def plot_diffusion_speed_up(self):
        speed_up_dict = {solver: {} for solver in solvers}
        fig, axs = plt.subplots(1, len(solvers))
        for solver, ax in zip(solvers, axs):

            for size, network in self.networks[solver].items():
                diff_speed_up = network.diffusion_speed_up()
                if diff_speed_up > 0:
                    speed_up_dict[solver][size] = diff_speed_up
                else:
                    speed_up_dict[solver][size] = np.NaN

            ax.plot(self.networks[solver].keys(), speed_up_dict[solver].values())

        directory = graph_dir() + os.path.sep + 'diffusion_speedup_' + '.pdf'
        print('Saving transition matrix in ', directory)
        fig.savefig(directory)


shape = 'SPT'
solvers = ['human', 'ant']
sizes = exp_types[shape]

if __name__ == '__main__':
    my_networks = Network_comparison.load_networks()
    my_network_comparison = Network_comparison(my_networks)
    my_network_comparison.plot_transition_matrices()
    my_network_comparison.plot_diffusion_speed_up()

    # TODO: Check whether the right solver number is involved (especially for human Medium).
    # TODO: Plot with equal state order
