from DataFrame.import_excel_dfs import dfs_human
from Directories import network_dir
import json
import os
import pandas as pd
from copy import copy


class Human:
    def __init__(self, filename, ts):
        self.filename = filename
        self.ts = ts

        self.checked = {}
        self.initial_network = None

    def calc(self):
        self.initial_network = copy(connectionMatrix)
        for state1, state2 in [('b1', 'f'), ('b2', 'f'), ('be', 'eb'), ('cg', 'g'), ('eg', 'g')]:
            self.checked[state1] = self.ts.count(state1)
            if self.checked[state1] > 0:
                self.initial_network.loc[state1][state2] = True
                self.initial_network.loc[state2][state1] = True


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
        state_series_dict = json.load(json_file)
        json_file.close()

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
    DEBUG = 1