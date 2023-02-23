import pandas as pd
from trajectory_inheritance.get import get
from Analysis.AttachmentStatistics.AntNumber_in_Chambers import ChamberCounter
import os
from Directories import network_dir
import json
from Setup.Maze import Maze
from plotly import express as px
from scipy.ndimage import gaussian_filter
import numpy as np
import json
from operator import truediv

chamberCount_fps = 1  # number of frames per second in chamber count time series

states = {'ab': [1, 0, 0], 'ac': [1, 0, 0],
          'e': [0, 0, 1], 'f': [0, 0.5, 0.5],
          'eb': [0.1, 0.9, 0], 'eg': [0.1, 0.9, 0],
          'b': [0.6, 0.4, 0], 'b1': [0.2, 0.5, 0.1], 'b2': [0.2, 0.5, 0.1],
          'be': [0.4, 0.6, 0], 'c': [0.5, 0.5, 0], 'cg': [0.1, 0.6, 0.2],
          'g': [0, 0.4, 0.6], 'h': [0, 0, 1], 'i': [0, 0, 1]}


class ChamberDensity(ChamberCounter):
    def __init__(self, x, ts=None):
        super().__init__(x, ts=ts)
        self.density = None
        self.circ_in_chambers = None

    def calc_density(self, circ_per_chamber: np.array):
        """
        Calculates the density of the shape in each chamber in cm^-1
        :param circ_per_chamber: circumference of the shape in each chamber in cm
        """
        def div(a, b):
            return [a1/b1 if b1 != 0 else None for a1, b1 in zip(a, b)]
        # divide elementwise the filtered_counts in each chamber by the circ_in_chambers
        self.density = {ch: div(self.filtered_counts[ch], circ_per_chamber[:, ch]) for ch in range(3)}

    def calc_circumference_in_chambers(self, circ):
        """
        Calculates the circumference of the shape in each chamber in cm (excluding the length close to the maze
        boundary)
        :param circ: circumference of the shape in cm
        """
        circ_per_time_step = [states[x] for x in self.ts]

        ch_raw = np.array(circ_per_time_step) * circ
        zero_circumference = ch_raw == 0

        ch0 = gaussian_filter(ch_raw[:, 0], sigma=20)
        ch1 = gaussian_filter(ch_raw[:, 1], sigma=20)
        ch2 = gaussian_filter(ch_raw[:, 2], sigma=20)

        ch0[zero_circumference[:, 0]] = 0
        ch1[zero_circumference[:, 1]] = 0
        ch2[zero_circumference[:, 2]] = 0

        # apply a gaussian smooth to smooth the data to axis 0
        self.circ_in_chambers = {0: ch0.tolist(), 1: ch1.tolist(), 2: ch2.tolist()}

    def corrected_density(self, d):

        d[np.array(self.ts[:len(d)]) == 'ab', 1:3] = np.nan
        d[np.array(self.ts[:len(d)]) == 'ac', 1:3] = np.nan
        d[np.array(self.ts[:len(d)]) == 'c', 2] = np.nan
        d[np.array(self.ts[:len(d)]) == 'e', 0] = np.nan
        d[np.array(self.ts[:len(d)]) == 'e', 2] = np.nan
        d[np.array(self.ts[:len(d)]) == 'eg', 0] = np.nan
        d[np.array(self.ts[:len(d)]) == 'eb', 2] = np.nan
        d[np.array(self.ts[:len(d)]) == 'f', 0] = np.nan
        d[np.array(self.ts[:len(d)]) == 'h', 0] = np.nan
        return d

    def calc_density_per_state(self):
        print(len(self.ts))
        print(len(self.filtered_counts[0]))
        # if abs((len(states_in_exp)/len(self.filtered_counts[0]) - 1)) > 0.0007:
        # raise ValueError('Number of states and number of frames... do they match?')

        state_densities = {state: {0: [], 1: [], 2: [], } for state in states}

        d = np.vstack(list(self.density.values())).transpose()
        d = self.corrected_density(d)

        for i, (state, ch_density) in enumerate(zip(self.ts, d.tolist())):
            for ch in range(3):
                state_densities[state][ch].append(ch_density[ch])

        return state_densities

    @classmethod
    def get_density_of_exp(cls, filename):
        x = get(filename)
        x.adapt_fps(chamberCount_fps)

        cc = ChamberDensity(x, ts=time_series_dict[filename])

        fc = x.find_fraction_of_circumference(ts=cc.ts)
        fc[fc < 0.15] = None
        circ_per_chamber = fc * Maze(x).circumference()

        cc.calc_raw_counts(non_attached=False)
        cc.calc_filtered_counts()

        cc.calc_density(circ_per_chamber)
        df = pd.DataFrame({'ch0': cc.density[0], 'ch1': cc.density[1], 'ch2': cc.density[2]})
        fig = px.line(df, x=x.frames, y=['ch0', 'ch1', 'ch2'])
        fig.show()
        # x.play(wait=100)

        density_per_state = cc.calc_density_per_state()
        av_dens_per_state = {state: {ch: mean(density_per_state[state][ch]) for ch in range(3)} for state in
                             states}

        return av_dens_per_state

    @classmethod
    def find_averages(cls, group):
        dfs = []
        for filename in group['filename']:
            results_directory = os.path.join('circumference_densities', filename + 'density_per_state.json')
            if not os.path.exists(results_directory):
                raise FileNotFoundError(results_directory + ' not found')

            with open(results_directory, 'r') as json_file:
                data = json.load(json_file)
                json_file.close()

            dfs.append(pd.DataFrame(data))
        multi_indexed = pd.concat(dfs, keys=list(group['filename']), axis=0)
        # find averages over first index in zipped
        av = multi_indexed.groupby(level=1).mean()
        std = multi_indexed.groupby(level=1).std()
        return av, std


def mean(l):
    if np.all(np.isnan(l)):
        return np.nan
    else:
        return np.nanmean(l)


df_exps = pd.read_excel('decent_tracking.xlsx', engine='openpyxl').dropna(subset=['size'])

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
    state_series_dict = json.load(json_file)
    json_file.close()


if __name__ == '__main__':
    # filename = 'L_SPT_4650014_LSpecialT_1_ants'
    # av_dens_per_state = ChamberDensity.get_density_of_exp(filename)
    # DEBUG = 1

    groups = df_exps.groupby('size')

    # for size, group in groups:
    #     print(size)
    #     for i, exp in group.iterrows():
    #         results_directory = os.path.join('circumference_densities', exp['filename'] + 'density_per_state.json')
    #         # if not os.path.exists(results_directory):
    #         av_dens_per_state = ChamberDensity.get_density_of_exp(exp['filename'])
    #
    #         # save in json file
    #         with open(results_directory, 'w') as json_file:
    #             json.dump(av_dens_per_state, json_file)
    #             json_file.close()

    for size, group in groups:
        print(size)
        av, std = ChamberDensity.find_averages(group)
        av.to_csv(os.path.join('circumference_densities', 'averages', str(size) + '_mean.csv'))
        std.to_csv(os.path.join('circumference_densities', 'averages', str(size) + '_std' + '.csv'))
