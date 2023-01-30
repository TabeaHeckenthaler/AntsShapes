from Analysis.Path_Length_States import *
from Analysis.average_carrier_number.averageCarrierNumber import averageCarrierNumber_dict

# time_series_dict_selected_states_sim, state_series_dict_selected_states_sim = Path.get_dicts(name='_selected_states_sim')

# myDataFrame = Altered_DataFrame(myDataFrame_sim)
# myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict_selected_states_sim)
# myDataFrame.df['state series'] = myDataFrame.df['filename'].map(state_series_dict_selected_states_sim)

myDataFrame = Altered_DataFrame(myDataFrame)
myDataFrame.df['food in back'] = df_food.df['food in back']
myDataFrame.df['time series'] = myDataFrame.df['filename'].map(time_series_dict)
myDataFrame.df['state series'] = myDataFrame.df['filename'].map(state_series_dict)
myDataFrame.df['average Carrier Number'] = myDataFrame.df['filename'].map(averageCarrierNumber_dict)

shape = 'SPT'
# sizes = exp_types[shape][solver]

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}

# def plot_all():
#     for solver in ['ant', 'human']:
#         dfss = myDataFrame.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver],
#         shape=shape, geometry=solver_geometry[solver], initial_cond='back')
#         for block_bool in [True, False]:
#             for size, dfs in dfss.items():
#                 for sep, df in dfs.items():
#                     print(size, sep)
#                     df['exp_day'] = df['filename'].map(exp_day)
#                     # df.sort_values(by='path length/minimal path length[]', inplace=True)
#                     df.sort_values(by='filename', inplace=True)
#
#                     fig, ax = plt.subplots()
#                     plot_bar_chart(df, ax, block=block_bool)
#                     save_fig(fig, size + sep + '_states_block_' + str(block_bool))


def plot_human(size='M (>7)', block=True):
    solver = 'human'
    dfss = myDataFrame.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver], shape=shape,
                                                geometry=solver_geometry[solver], initial_cond='back')
    fig, axs = plt.subplots(2, sharex='col', figsize=(7, 10),
                            gridspec_kw={'height_ratios': [len(dfss[size]['communication']),
                                                           len(dfss[size]['non_communication'])]})

    create_bar_chart(dfss[size]['communication'], axs[0], block=block)
    create_bar_chart(dfss[size]['non_communication'], axs[1], block=block)
    plt.subplots_adjust(hspace=.0)
    save_fig(fig, 'human_' + size + str(block))


def plot_ant(size='S (> 1)', block=True):
    solver = 'ant'
    dfss = myDataFrame.get_separate_data_frames(solver=solver,
                                                plot_separately=plot_separately[solver], shape=shape,
                                                geometry=solver_geometry[solver],
                                                initial_cond='back')
    fig, axs = plt.subplots(2, sharex='col', figsize=(7, 10),
                            gridspec_kw={'height_ratios': [len(dfss[size]['winner']), len(dfss[size]['looser'])]})

    create_bar_chart(dfss[size]['winner'], axs[0], block=block)
    create_bar_chart(dfss[size]['looser'], axs[1], block=block)
    plt.subplots_adjust(hspace=.0)
    save_fig(fig, 'ant_' + size + str(block))


def plot_pheidole(size='S (> 1)', block=True):
    solver = 'ant'
    dfss = myDataFrame.get_separate_data_frames(solver=solver,
                                                plot_separately=plot_separately[solver], shape=shape,
                                                geometry=solver_geometry[solver],
                                                initial_cond='back')
    fig, axs = plt.subplots(2, sharex='col', figsize=(7, 10),
                            gridspec_kw={'height_ratios': [len(dfss[size]['winner']), len(dfss[size]['looser'])]})

    create_bar_chart(dfss[size]['winner'], axs[0], block=block)
    create_bar_chart(dfss[size]['looser'], axs[1], block=block)
    plt.subplots_adjust(hspace=.0)
    save_fig(fig, 'ant_' + size + str(block))

#
# def plot_sim(block=True):
#     fig, axs = plt.subplots(2, sharex='col', figsize=(7, 10),
#                             gridspec_kw={'height_ratios': [myDataFrame_sim['winner'].sum(),
#                                                            len(myDataFrame_sim['winner']) -
#                                                            myDataFrame_sim['winner'].sum()]})
#     plot_bar_chart(myDataFrame_sim[myDataFrame_sim['winner']], axs[0], block=block)
#     plot_bar_chart(myDataFrame_sim[~myDataFrame_sim['winner']], axs[1], block=block)
#     plt.subplots_adjust(hspace=.0)
#     save_fig(fig, 'sim_' + 'M' + str(block))


if __name__ == '__main__':
    block = [False]
    for b in block:
        # plot_sim(block=b)
        plot_ant('Single (1)', block=b)
        plot_ant('S (> 1)', block=b)
        plot_ant('XL', block=b)
        plot_ant('M', block=b)
        plot_ant('L', block=b)

        plot_pheidole('Single (1)', block=b)
        plot_pheidole('S (> 1)', block=b)
        plot_pheidole('XL', block=b)
        plot_pheidole('M', block=b)
        plot_pheidole('L', block=b)

        plot_human('Large', block=b)
        plot_human('M (>7)', block=b)
        plot_human('Small', block=b)

