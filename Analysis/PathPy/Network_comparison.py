from matplotlib import pyplot as plt
from trajectory_inheritance.exp_types import solver_geometry, exp_types
from Analysis.PathPy.Paths import PathWithoutSelfLoops, PathsTimeStamped, plot_seperately
from Analysis.PathPy.Network import Network
from Analysis.GeneralFunctions import graph_dir
from Analysis.PathPy.SPT_states import states, forbidden_transition_attempts, allowed_transition_attempts
import os
from DataFrame.Altered_DataFrame import Altered_DataFrame
from Analysis.GeneralFunctions import flatten
import numpy as np


def flatten_dict(dict1):
    new = {}
    for key1, dict2 in dict1.items():
        for key2, value in dict2.items():
            new[str(key1) + ' ' + str(key2)] = value
    return new


class Network_comparison:
    def __init__(self, networks: dict = None):
        self.networks = networks

    @classmethod
    def load_networks(cls, only_states=True, Path_class: type = PathWithoutSelfLoops, symmetric_states=False) -> dict:
        networks = {solver: {} for solver in solvers}
        for solver in solvers:
            df = Altered_DataFrame()
            dfs = df.get_separate_data_frames(solver, plot_seperately[solver], shape=shape)
            networks[solver] = {}
            if type(list(dfs.values())[0]) is dict:
                dfs = flatten_dict(dfs)
            for key, df in dfs.items():
                print(key)
                paths = Path_class(solver, shape, solver_geometry[solver])
                paths.load_paths(filenames=df.filename, only_states=only_states, symmetric_states=symmetric_states)
                my_network = Network.init_from_paths(paths, solver, shape)
                # we could also do: get results, if we are sure that we saved the right thing
                my_network.markovian_analysis()
                networks[solver][key] = my_network
        return networks

    def all_networks(self):
        return flatten([list(v1) for v1 in [v.values() for v in self.networks.values()]])

    def plot_transition_matrices(self, scale='log'):
        # state_order = states[1:] + allowed_transition_attempts + forbidden_transition_attempts
        state_order = sorted(set(flatten([n.T.columns for n in self.all_networks()])))

        for solver in self.networks.keys():
            fig, axs = plt.subplots(2, np.ceil(len(self.networks[solver])/2).astype(int))
            if axs.ndim != 1:
                axs = flatten(axs)
            for (size, network), ax in zip(self.networks[solver].items(), axs):
                network.plot_transition_matrix(title=size, axis=ax, state_order=state_order, scale='log')

            directory = graph_dir() + os.path.sep + 'transition_matrix_' + solver + '.pdf'
            print('Saving transition matrix in ', directory)
            plt.subplots_adjust(wspace=0.01, hspace=0.3)
            fig.savefig(directory)

    def plot_diffusion_speed_up(self):
        speed_up_dict = {solver: {} for solver in plot_seperately}
        fig, axs = plt.subplots(1, len(plot_seperately))
        for solver, ax in zip(plot_seperately, axs):

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
solvers = ['humanhand', 'human', 'ant']  # add humanhand
sizes = exp_types[shape]
sizes['human'] = ['Large', 'Medium', 'Small']

if __name__ == '__main__':
    my_networks = Network_comparison.load_networks(only_states=False, Path_class=PathsTimeStamped,
                                                   symmetric_states=True)
    my_network_comparison = Network_comparison(my_networks)
    my_network_comparison.plot_transition_matrices()
    # my_network_comparison.plot_diffusion_speed_up()

    # TODO: Check whether the right solver number is involved (especially for human Medium).
    # TODO: Plot with equal state order
