from DataFrame.import_excel_dfs import df_ant_excluded
import pandas as pd
import matplotlib.pyplot as plt
from colors import colors_state
from DataFrame.import_excel_dfs import df_ant_excluded
import json
from os import path
from Directories import network_dir, home
from itertools import groupby
from trajectory_inheritance.get import get
from tqdm import tqdm
import numpy as np

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

with open(path.join(home, 'DataFrame\\frameNum_dict.json'), 'r') as json_file:
    frameNum_dict = json.load(json_file)

df_ant_excluded['time series'] = df_ant_excluded['filename'].map(time_series_dict)
df_ant_excluded['frameNum'] = df_ant_excluded['filename'].map(frameNum_dict)
ant_sizes = ['Single (1)', 'S (> 1)', 'M', 'L', 'XL'][1:]

chosen_indices_cg = {}


# convert seconds to string with minutes:seconds
def sec_to_min_sec(sec):
    minutes = int(sec / 60)
    seconds = int(sec - minutes * 60)
    # add leading zero
    if seconds < 10:
        seconds = '0' + str(seconds)
    return str(minutes) + ':' + str(seconds)


def extend_time_series_to_match_frames(ts, frame_len):
    indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                         for _ in range(frame_len)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


# sort by size
for size in ant_sizes:
    df = df_ant_excluded[df_ant_excluded['size'] == size]
    for i, row in tqdm(list(df.iterrows()), total=len(df), desc=size):
        filename = row['filename']
        # filename = 'L_SPT_4650005_LSpecialT_1_ants (part 1)'
        time_series = time_series_dict[filename]
        x = get(filename)
        frameNum = len(x.frames)
        time_series = extend_time_series_to_match_frames(time_series, frameNum)
        vel = np.linalg.norm(x.velocity(smoothed=1), axis=1)
        x.play(ts=time_series)

        # indices_c = [i for i, state in enumerate(time_series) if state == 'c']
        indices_cg = [i for i, state in enumerate(time_series) if state == 'cg']
        if len(indices_cg) > 100:
            # inds = [indices_cg[i] for i in range(0, len(indices_cg), int(len(indices_cg) / 3))]
            # choose 3 indices at random
            inds = np.random.choice(indices_cg, 2, replace=False)
            # sort inds by size
            inds = sorted(inds, key=lambda i: vel[i])
            chosen_indices_cg[filename + '_0'] = x.frames[inds[0]]
            chosen_indices_cg[filename + '_1'] = x.frames[inds[1]]
            # chosen_indices_cg[filename + '_2'] = x.frames[inds[2]]

# write chosen_indices_cg to dataframe
df = pd.DataFrame.from_dict(chosen_indices_cg, orient='index')
# sort df by filename
df = df.sort_index()

# save as excel file
df.to_excel('ant_counting_prep.xlsx')
