from DataFrame.dataFrame import myDataFrame
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.PathLength.PathLength import path_length_dict
from Analysis.PathPy.Path import time_series_dict, state_series_dict, Path
from trajectory_inheritance.exp_types import solver_geometry
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import pandas as pd


myDataFrame['minimal path length [length unit]'] = myDataFrame['filename'].map(minimal_path_length_dict)
myDataFrame['path length [length unit]'] = myDataFrame['filename'].map(path_length_dict)
myDataFrame['path length/minimal path length[]'] = myDataFrame['path length [length unit]'] / \
                                                   myDataFrame['minimal path length [length unit]']
myDataFrame['time series'] = myDataFrame['filename'].map(time_series_dict)
myDataFrame['state series'] = myDataFrame['filename'].map(state_series_dict)


df = myDataFrame[(myDataFrame['size'] == 'L') &
                 (myDataFrame['shape'] == 'SPT') &
                 (myDataFrame['maze dimensions'] == solver_geometry['ant'][0]) &
                 (myDataFrame['initial condition'] == 'back')
                ]


def exp_day(filename):
    return filename.split('_')[2]


def time_spent_in_states(state_series, time_step=0.25):
    state_series = Path.only_states(state_series)
    state_series = Path.symmetrize(state_series)
    d = dict()

    for state in set(state_series):
        d[state] = state_series.count(state) * time_step
    return d


if __name__ == '__main__':
    filename = 'L_SPT_4660011_LSpecialT_1_ants'
    # print(time_series_dict[filename])
    # p = Path(time_step=0.25, time_series=time_series_dict[filename])
    # p.bar_chart()

    df['exp_day'] = df['filename'].map(exp_day)
    df.sort_values(by='path length/minimal path length[]', inplace=True)
    df.sort_values(by='filename', inplace=True)

    fig, ax = plt.subplots()
    for filename in df['filename']:
        p = Path(time_step=0.25, time_series=time_series_dict[filename])
        p.bar_chart(ax=ax, axis_label=filename)

    dict_b = {}
    df['state dict'] = df['filename'].map(dict_b)
    for i, exp in df.iterrows():
        print(exp['filename'])
        dict_b[exp['filename']] = time_spent_in_states(exp['time series'])

    DEBUG = 1
    states_df = pd.DataFrame(df['state dict'].tolist(), index=df.index)
    ax = states_df.plot(kind='bar', stacked=True, title='states')
    ax.set_xticklabels(df['exp_day'])
    ax.scatter(df['exp_day'], df['winner'].astype(int)*3500, color='green')
    plt.show()
    DEBUG = 1

    for key in df['filename']:
        if time_series_dict[key] is not None:
            plt.plot(Path.only_states(time_series_dict[key]))

    fig, ax = plt.subplots()
    df['time in c [s]'] = None
    df.plot(x='time in c [s]', y='path length/minimal path length[]', marker='.', linewidth=0, ax=ax)

    for index, v in df.iterrows():
        ax.annotate(index, [v['time in c [s]'], v['path length/minimal path length[]']])

    DEBUG = 1
