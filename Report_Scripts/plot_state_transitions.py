from Analysis.Path_Length_States import *


myDataFrame = Altered_DataFrame(myDataFrame)
myDataFrame.df['food in back'] = df_food.df['food in back']

myDataFrame.df['minimal path length [length unit]'] = myDataFrame.df['filename'].map(minimal_path_length_dict)
myDataFrame.df['path length [length unit]'] = myDataFrame.df['filename'].map(path_length_dict)
myDataFrame.df['path length/minimal path length[]'] = myDataFrame.df['path length [length unit]'] / \
                                                      myDataFrame.df['minimal path length [length unit]']
myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict_selected_states)
myDataFrame.df['state series'] = myDataFrame.df['filename'].map(state_series_dict_selected_states)

solver = 'ant'
shape = 'SPT'
sizes = exp_types[shape][solver]

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}
dfss = myDataFrame.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver], shape=shape,
                                            geometry=solver_geometry[solver], initial_cond='back')

for size, dfs in dfss.items():
    for sep, df in dfs.items():
        print(size, sep)
        if size == 'L':
            df['exp_day'] = df['filename'].map(exp_day)
            df.sort_values(by='path length/minimal path length[]', inplace=True)
            df.sort_values(by='filename', inplace=True)

            fig, ax = plt.subplots()
            create_bar_chart(df, ax, block=True)
            save_fig(fig, size + sep + '_states')

# in_state_chart()
DEBUG = 1
