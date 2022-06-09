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
from Analysis.PathPy.diffusion_time import DiffusionTime
from Analysis.PathPy.Path import time_step
import pandas as pd
from matplotlib.pyplot import cm


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
                # if key == 'XL looser':
                #     DEBUG = 2
                paths = Path_class(solver, shape, solver_geometry[solver])
                paths.load_paths(filenames=df.filename, only_states=only_states, symmetric_states=symmetric_states)
                my_network = Network.init_from_paths(paths, solver, shape)
                # we could also do: get results, if we are sure that we saved the right thing
                if len(paths.nodes) > 0:
                    my_network.markovian_analysis()
                    networks[solver][key] = my_network
        return networks

    def all_networks(self):
        return flatten([list(v1) for v1 in [v.values() for v in self.networks.values()]])

    def get_state_order(self):
        states_ = set(flatten([n.T.columns for n in self.all_networks()]))
        # state_order = sorted(set(flatten([n.T.columns for n in self.all_networks()])))
        # alphabet_string = 'bacdefghi'
        alphabet_string = 'bf b be ba ab a ac ca c cg ' \
                          'ce cd ec dc e d ef df fe fd ' \
                          'f fb fh hf h i'.split(' ')
        # alphabet = {c: i for i, c in enumerate(alphabet_string)}
        # state_order = sorted(states_, key=lambda word: [alphabet.get(c, ord(c)) for c in word])
        state_order = [a for a in alphabet_string if a in states_]
        return state_order

    def plot_transition_matrices(self, scale='log'):
        # state_order = states[1:] + allowed_transition_attempts + forbidden_transition_attempts
        state_order = self.get_state_order()

        for solver in self.networks.keys():
            ncols = 2
            n_networks = len(self.networks[solver])
            fig, axs = plt.subplots(np.ceil(n_networks/ncols).astype(int), min(ncols, n_networks))
            if axs.ndim != 1:
                axs = flatten(axs)
            for (size, network), ax in zip(self.networks[solver].items(), axs):
                network.plot_transition_matrix(title=size, axis=ax, state_order=state_order, scale=scale)

            directory = graph_dir() + os.path.sep + 'transition_matrix_' + solver + '.pdf'
            print('Saving transition matrix in ', directory)
            fig.set_figheight(np.ceil(n_networks/ncols).astype(int) * 2.3)
            fig.set_figwidth(ncols * 2)
            # plt.subplots_adjust(wspace=0.01, hspace=0.3)
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

    def calc_diffusion_times(self):
        diffusion_time = {solver: {} for solver in plot_seperately}
        for solver in plot_seperately:
            for size, n in self.networks[solver].items():
                print(size)
                if 'looser' in size:
                    pass
                else:
                    if 'winner' in size:
                        n_plot = Network.init_from_paths(n.paths + self.networks[solver][size.replace('winner', 'looser')].paths,
                                                        solver, shape, size)
                    else:
                        n_plot = n
                    diff_time_calc = DiffusionTime(solver, shape, solver_geometry[solver], network=n_plot,
                                                   time_step=time_step)
                    diffusion_time[solver][size] = diff_time_calc.calculate_diffusion_time()
        return diffusion_time

    def plot_diffusion_times(self, diffusion_time):
        fig, axs = plt.subplots(1, len(plot_seperately))
        state_order = self.get_state_order()

        for solver, ax in zip(plot_seperately, axs):
            ax.set_title(solver)
            color = dict(zip(self.networks[solver].keys(),
                             cm.rainbow(np.linspace(0, 1, len(self.networks[solver].keys())))))

            for size, series in diffusion_time[solver].items():
                missing = pd.Series({s: np.NaN for s in state_order if s not in series.index})
                series = series.append(missing)
                series = series.loc[state_order]
                ax.plot(series, label=size.split(' winner')[0], color=color[size], linewidth=2)

            ax.legend()
        axs[0].set_ylabel('absorption time [s]')
        directory = graph_dir() + os.path.sep + 'diffusion_time' + '.pdf'
        print('Saving diffusion times in ', directory)
        plt.show()
        fig.savefig(directory)


shape = 'SPT'
solvers = ['humanhand', 'human', 'ant']  # add humanhand
sizes = exp_types[shape]
sizes['human'] = ['Large', 'Medium', 'Small']

if __name__ == '__main__':
    my_networks = Network_comparison.load_networks(only_states=False, Path_class=PathsTimeStamped,
                                                   symmetric_states=True)
    my_network_comparison = Network_comparison(my_networks)
    diffusion_time = my_network_comparison.calc_diffusion_times()
    my_network_comparison.plot_diffusion_times(diffusion_time)
    plt.show()
    # my_network_comparison.plot_transition_matrices(scale='log')
    DEBUG = 1

    # my_network_comparison.plot_diffusion_speed_up()

    # TODO: Check whether the right solver number is involved (especially for human Medium).
    # TODO: Plot with equal state order
