import random

import numpy as np
from lempel_ziv_complexity import lempel_ziv_complexity
from DataFrame.import_excel_dfs import dfs_human, dfs_ant
from Directories import network_dir
from os import path
import json
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
to_single_character = {'b': '0', 'be': '1', 'b1': '2', 'b2': '3',
                       'ac': '4', 'ab': '5', 'c': '6', 'cg': '7',
                       'e': '8', 'eb': '9', 'eg': 'a', 'f': 's', 'g': 'd',
                       'h': 'f', 'i': 'g'}


class Lempel_Ziv:

    @staticmethod
    def rename(l):
        l = [to_single_character[x] for x in l]
        return l

    @staticmethod
    def calc():
        lempel_ziv_complexity_dict = {}
        length = {}

        for dfs_solver in [dfs_human, dfs_ant]:
            for key, df in dfs_solver.items():
                print(key)
                for filename in df['filename']:
                    time_series = ''.join(Lempel_Ziv.rename(time_series_dict[filename]))
                    lz_complexity = lempel_ziv_complexity(time_series)

                    lempel_ziv_complexity_dict[filename] = lz_complexity
                    length[filename] = len(time_series)
        return lempel_ziv_complexity_dict, length

    @staticmethod
    def save(lempel_ziv_complexity_dict, length):
        # save to json
        with open(path.join(network_dir, 'lempel_ziv_complexity.json'), 'w') as json_file:
            json.dump(lempel_ziv_complexity_dict, json_file)
            json_file.close()

        with open(path.join(network_dir, 'length.json'), 'w') as json_file:
            json.dump(length, json_file)
            json_file.close()

    @staticmethod
    def load():
        with open(path.join(network_dir, 'lempel_ziv_complexity.json'), 'r') as json_file:
            lempel_ziv_complexity_dict = json.load(json_file)
            json_file.close()

        with open(path.join(network_dir, 'length.json'), 'r') as json_file:
            length = json.load(json_file)
            json_file.close()
        return lempel_ziv_complexity_dict, length

    @staticmethod
    def plot(lempel_ziv_complexity_dict, length):
        averages = {}
        std = {}
        plt.figure(figsize=(10, 10))

        plt.title('Lempel-Ziv Complexity')

        for dfs_solver in [dfs_human, dfs_ant]:
            for key, df in dfs_solver.items():
                df = df[df['winner']]
                df['lzc'] = df['filename'].map(lempel_ziv_complexity_dict)
                df['length'] = df['filename'].map(length)
                df['lzc_norm'] = df['lzc'] / df['length']
                averages[key] = df['lzc_norm'].mean()
                std[key] = df['lzc_norm'].std() / np.sqrt(len(df['lzc_norm']))
                df['lzc_norm'].hist(bins=10, histtype='step', density=True, cumulative=True, label=key)
        plt.legend()

        plt.errorbar(averages.keys(), averages.values(), yerr=std.values(), fmt='o')
        # set y label as 'Lempel-Ziv Complexity/length of string'
        plt.ylabel('Lempel-Ziv Complexity/length of string')
        plt.show()

        DEBUG = 1


if __name__ == "__main__":
    with open(path.join(network_dir, 'state_series_selected_states.json'), 'r') as file:
        time_series_dict = json.load(file)
        file.close()

    lzc_dict, length_dict = Lempel_Ziv.calc()
    Lempel_Ziv.save(lzc_dict, length_dict)
    lzc_dict, length_dict = Lempel_Ziv.load()
    Lempel_Ziv.plot(lzc_dict, length_dict)

# generate random sequence of letters
# import random
# import string

# random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(1000000))