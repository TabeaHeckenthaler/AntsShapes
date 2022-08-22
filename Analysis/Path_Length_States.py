from DataFrame.dataFrame import myDataFrame
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.PathLength.PathLength import path_length_dict
from Analysis.PathPy.Path import time_series_dict, state_series_dict
from trajectory_inheritance.exp_types import solver_geometry
from matplotlib import pyplot as plt
import pandas as pd


myDataFrame['minimal path length [length unit]'] = myDataFrame['filename'].map(minimal_path_length_dict)
myDataFrame['path length [length unit]'] = myDataFrame['filename'].map(path_length_dict)
myDataFrame['path length/minimal path length[]'] = myDataFrame['path length [length unit]'] / \
                                                   myDataFrame['minimal path length [length unit]']
myDataFrame['time series'] = myDataFrame['filename'].map(time_series_dict)
myDataFrame['state series'] = myDataFrame['filename'].map(state_series_dict)


df = myDataFrame[(myDataFrame['size'] == 'L') &
                 (myDataFrame['shape'] == 'SPT') &
                 (myDataFrame['maze dimensions'] == solver_geometry['ant'][0])]


def time_spent_in_state(time_series, state='c'):
    time_series = [l[0] for l in time_series]
    time_step = 0.25
    return time_series.count(state) * time_step


def time_spent_in_states(time_series):
    time_series = [l[0] for l in time_series]
    d = dict()
    for state in set(time_series):
        d[state] = time_spent_in_state(time_series, state=state)
    return d

# time_spent_in_states(df.loc[153]['time series'])

dict_b = {}
for i, exp in df.iterrows():
    print(exp['filename'])
    dict_b[exp['filename']] = time_spent_in_states(exp['time series'])

df['state dict'] = df['filename'].map(dict_b)
df.sort_values(by='path length/minimal path length[]', inplace=True)
df.sort_values(by='filename', inplace=True)

states_df = pd.DataFrame(df['state dict'].tolist(), index=df.index)
states_df.plot(kind='bar', stacked=True, title='states')

fig, ax = plt.subplots()
df['time in c [s]'] = None
df.plot(x='time in c [s]', y='path length/minimal path length[]', marker='.', linewidth=0, ax=ax)

for index, v in df.iterrows():
    ax.annotate(index, [v['time in c [s]'], v['path length/minimal path length[]']])

DEBUG = 1
