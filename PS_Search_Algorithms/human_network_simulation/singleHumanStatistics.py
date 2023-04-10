from DataFrame.import_excel_dfs import dfs_human
from Directories import network_dir, home
import json
import os
import pandas as pd
from copy import copy

connectionMatrix = pd.read_excel(
    home + "\\PS_Search_Algorithms\\human_network_simulation\\ConnectionMatrix.xlsx",
    index_col=0, dtype=bool)

false_connections = [('b1', 'f'), ('b2', 'f'), ('be', 'eb'), ('cg', 'g'), ('eg', 'g')]


class Human:
    def __init__(self, filename, ts):
        self.filename = filename
        self.ts = ts

        self.checked = {}
        self.initial_network = None

    def calc(self):
        self.initial_network = copy(connectionMatrix)
        for state1, state2 in false_connections:
            self.checked[state1] = self.ts.count(state1)
            if self.checked[state1] > 0:
                self.initial_network.loc[state1][state2] = True
                self.initial_network.loc[state2][state1] = True

    def first_choices(self):
        first_choices = {}
        for junction in connectionMatrix.index:
            if junction in self.ts:
                first_choices[junction] = self.ts[self.ts.index(junction) + 1]
        return first_choices

    @staticmethod
    def calc_first_passage_affinity():
        """
        How affine where humans at every junction to go to every neighboring node.
        :return:
        """
        # Load all possible connections
        affinity = pd.DataFrame(0, index=connectionMatrix.index, columns=connectionMatrix.columns)

        # iterate over all experiments and look at the first choice at every junction
        df_small = dfs_human['Small']
        df_small['state series'] = df_small['filename'].map(state_series_dict)

        first_choices = []
        for filename, ts in zip(df_small['filename'], df_small['state series']):
            human = Human(filename, ts)
            first_choices.append(human.first_choices())

        # iterate over all junctions and add up the first choices
        for first_choice in first_choices:
            for junction, choice in first_choice.items():
                affinity.loc[junction][choice] += 1

        # normalize every junction to 1.
        affinity = affinity.div(affinity.sum(axis=1), axis=0)
        # save the affinity matrix
        affinity.to_excel('first_passage_affinity_single_human.xlsx')

    def first_passage_probability_estimation(self):
        pass
        # load the distance matrix
        distances = pd.read_excel("calc_distances.xlsx", index_col=0)

        # divide every first_passage affinitiy by the distance matrix



        # normalize every node to 1.

    @staticmethod
    def old_initial_connection_network_calculation():
        df_small = dfs_human['Small']
        df_small['state series'] = df_small['filename'].map(state_series_dict)
        connectionMatrix = pd.read_excel("ConnectionMatrix.xlsx", index_col=0).astype(bool)

        initialNetworks = {}

        for filename, ts in zip(df_small['filename'], df_small['state series']):
            human = Human(filename, ts)
            human.calc()
            initialNetworks[filename] = human.initial_network

        # find the average of all initialNetworks
        initialNetworks_mean = pd.concat(initialNetworks).astype(float).groupby(level=1).mean()
        initialNetworks_mean.to_excel('small_average_initialNetworks.xlsx')


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
        state_series_dict = json.load(json_file)
        json_file.close()

    # before talking to Ofer
    # Human.old_initial_connection_network_calculation()

    # after talking to Ofer
    Human.calc_first_passage_affinity()
    DEBUG = 1