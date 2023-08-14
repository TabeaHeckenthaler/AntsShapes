from trajectory_inheritance.get import get
import os
from Directories import data_home
import pandas as pd
import json
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from tqdm import tqdm
from itertools import groupby

# This has all moved to C:\Users\tabea\PycharmProjects\AntsShapes\DataFrame\gillespie_dataFrame.py!!!!

date = '2023_06_27'


def make_df():
    # make a dataFrame
    df = pd.DataFrame(columns=['filename', 'size'])
    super_dir = os.path.join(data_home, 'Pickled_Trajectories', 'Gillespie_Trajectories', date)
    for d in os.listdir(super_dir):
        filenames = os.listdir(os.path.join(super_dir, d))
        print(filenames)
        df = df.append(pd.DataFrame({'filename': filenames, 'size': d}), ignore_index=True)

    # to excel file
    df.to_excel(date + '_sim.xlsx')


def calc_time_series(x, conf_space_labeled, time_step=0.01):
    coords_in_cs = [coords for coords in x.iterate_coords_for_ps(time_step=time_step)]
    indices = [conf_space_labeled.coords_to_indices(*coords, shape=conf_space_labeled.space_labeled.shape)
               for coords in coords_in_cs]
    labels = [None]
    for i, index in enumerate(indices):
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            label = conf_space_labeled.find_closest_state(index)
        labels.append(label)
    labels = labels[1:]
    return labels


def time_stamped_series(time_series, time_step) -> list:
    groups = groupby(time_series)
    return [(label, sum(1 for _ in group) * time_step) for label, group in groups]


def get_time_series(cs_labeled):
    ts = {}
    df = pd.read_excel(date + '_sim.xlsx')
    for filename in tqdm(df['filename']):
        x = get(filename)
        x.position = x.position * {'XL': 1/4, 'L': 1/2, 'M': 1, 'S': 2, 'XS': 4}[df[df['filename'] == filename]['size'].iloc[0]]
        x.adapt_fps(4)
        ts[filename] = calc_time_series(x, cs_labeled, time_step=1/x.fps)

    with open(date + '_sim_time_series.json', 'w') as f:
        json.dump(ts, f)


if __name__ == '__main__':
    # make_df()
    geometry = ('MazeDimensions_new2021_SPT_ant.xlsx',
                'LoadDimensions_new2021_SPT_ant.xlsx')
    cs_labeled = ConfigSpace_SelectedStates('ant', 'M', 'SPT', geometry)
    cs_labeled.load_final_labeled_space()
    get_time_series(cs_labeled)
