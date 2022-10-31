from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.Efficiency.PathLength import path_length_dict
from Analysis.PathPy.Path import time_series_dict, state_series_dict, time_series_dict_selected_states, \
    state_series_dict_selected_states, Path
from trajectory_inheritance.exp_types import solver_geometry, exp_types
from matplotlib import pyplot as plt
from trajectory_inheritance.get import get
import pandas as pd
from DataFrame.plot_dataframe import save_fig
from DataFrame.Altered_DataFrame import Altered_DataFrame, myDataFrame_sim
from Analysis.average_carrier_number.averageCarrierNumber import myDataFrame
from DataFrame.food_in_the_back import myDataFrame as df_food


def exp_day(filename):
    return '_'.join(filename.split('_')[:3])


def time_spent_in_states(state_series, time_step=0.25):
    state_series = Path.only_states(state_series)
    state_series = Path.symmetrize(state_series)
    d = dict()

    for state in set(state_series):
        d[state] = state_series.count(state) * time_step
    return d


def create_bar_chart(df, ax, block=False, sorted=True):
    if sorted:
        df = df.sort_values('time [s]')

    # for filename, ts, winner, food in zip(df['filename'], df['time series'], df['winner'], df['food in back']):
    for filename, ts in zip(df['filename'], df['time series']):
        p = Path(time_step=0.25, time_series=ts)
        print(filename)
        p.bar_chart(ax=ax, axis_label=exp_day(filename), block=block)
        if not block:
            ax.set_xlabel('time [min]')
        else:
            ax.set_xlabel('')
        # ax.set_xlim([0, 20])
    DEBUG = 1


def in_state_chart():
    """
    Not sure about this function
    """
    dict_b = {}
    df['state dict'] = df['filename'].map(dict_b)
    for i, exp in df.iterrows():
        print(exp['filename'])
        dict_b[exp['filename']] = time_spent_in_states(exp['time series'])
    states_df = pd.DataFrame(df['state dict'].tolist(), index=df.index)
    ax = states_df.plot(kind='bar', stacked=True, title='states')
    ax.set_xticklabels(df['exp_day'])
    ax.scatter(df['exp_day'], df['winner'].astype(int) * 3500, color='green')
    plt.show()

    fig, ax = plt.subplots()
    df['time in c [s]'] = None
    df.plot(x='time in c [s]', y='path length/minimal path length[]', marker='.', linewidth=0, ax=ax)

    for index, v in df.iterrows():
        ax.annotate(index, [v['time in c [s]'], v['path length/minimal path length[]']])


if __name__ == '__main__':
    # filename = 'L_SPT_4660011_LSpecialT_1_ants'
    # print(time_series_dict[filename])
    # p = Path(time_step=0.25, time_series=time_series_dict[filename])
    # p.bar_chart()
    myDataFrame = Altered_DataFrame(myDataFrame)
    DEBUG = 1
    myDataFrame.df['food in back'] = df_food.df['food in back']

    myDataFrame.df['minimal path length [length unit]'] = myDataFrame.df['filename'].map(minimal_path_length_dict)
    myDataFrame.df['path length [length unit]'] = myDataFrame.df['filename'].map(path_length_dict)
    myDataFrame.df['path length/minimal path length[]'] = myDataFrame.df['path length [length unit]'] / \
                                                          myDataFrame.df['minimal path length [length unit]']
    myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict)
    myDataFrame.df['state series'] = myDataFrame.df['filename'].map(state_series_dict)

    solver = 'ant'
    shape = 'SPT'
    sizes = exp_types[shape][solver]

    plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}
    dfss = myDataFrame.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver], shape=shape,
                                                geometry=solver_geometry[solver], initial_cond='back')

    for size, dfs in dfss.items():
        for sep, df in dfs.items():
            print(size, sep)

            df['exp_day'] = df['filename'].map(exp_day)
            df.sort_values(by='path length/minimal path length[]', inplace=True)
            df.sort_values(by='filename', inplace=True)

            fig, ax = plt.subplots()
            create_bar_chart(df, ax)
            save_fig(fig, size + sep + '_states')
    # in_state_chart()
    DEBUG = 1
