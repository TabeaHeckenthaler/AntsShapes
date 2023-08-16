from Directories import home
import pandas as pd
import json
import numpy as np
from trajectory_inheritance.get import get

states = ['ab', 'ac', 'b', 'be1', 'be2', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']

df_human = pd.read_excel(home + '\\DataFrame\\final\\df_human.xlsx')

# df_special should contain only the large human that are not communicating
df_special = df_human[(df_human['size'] == 'Large') & (df_human['communication'] == True)]
# for every state in states put value into df_special for every filename

with open(home + '\\ConfigSpace\\time_series_human.json', 'r') as json_file:
    time_series_human = json.load(json_file)
    json_file.close()

with open('human_pathlengths_all_states.json', 'r') as f:
    json_file = json.load(f)
filenames = json_file.keys()


def translation_in_state(filename, state, json_file):
    return np.sum(list(json_file[filename][state][0].values()))


def rotation_in_state(filename, state, json_file):
    return np.sum(list(json_file[filename][state][1].values()))


def calc_trans_rot_per_state_per_filename(filenames, json_file):
    translation_in_states = {}
    rotation_in_states = {}
    for state in states:
        translation_in_states[state] = \
            {filename: translation_in_state(filename, state, json_file) for filename in filenames}
        rotation_in_states[state] = \
            {filename: rotation_in_state(filename, state, json_file) for filename in filenames}
    return translation_in_states, rotation_in_states


translation_in_states, rotation_in_states = calc_trans_rot_per_state_per_filename(filenames, json_file)

for state in states:
    for filename in filenames:
        df_special.loc[df_special['filename'] == filename, state] = translation_in_states[state][filename]
        df_special.loc[df_special['filename'] == filename, state + '_rot'] = rotation_in_states[state][filename]

# find means of columns b1 and b2
mean_b1 = df_special['b1'].mean()
mean_b2 = df_special['b2'].mean()
mean_be1 = df_special['be1'].mean()
mean_be2 = df_special['be2'].mean()

df_special_weird1 = df_special[((df_special['b1'] + df_special['b2']) < 0.5) & (df_special['b'] > 1)]


for filename in df_special_weird1['filename']:
    traj = get(filename)
    traj.play(ts=time_series_human[filename])

# observations: human large NC always enter b1/b2. Often don't try out a lot, instead stay mostly in the middle.
#               human large NC when entering b walk to the end (meaning b1 and b2). But, mostly b2 is counted,
#               because the calibration of the camera slightly biases the position of the shape towards b2.

# observations: human large C often will enter b, and walk to the middle and then turn to be1 or be2.
#               They often do not walk until the end.
# TODO: Percent that entered b, but did not walk in b2 or b1
