
# This has all moved to C:\Users\tabea\PycharmProjects\AntsShapes\DataFrame\gillespie_dataFrame.py!!!!


# from Analysis.PathPy.Path import *
# from Directories import home
#
#
# add = '_gillespie'
# # with open(os.path.join(network_dir, 'time_series_selected_states' + add + '.json'), 'r') as json_file:
# #     time_series_dict = json.load(json_file)
# #     json_file.close()
# #
# # with open(os.path.join(network_dir, 'state_series_selected_states' + add + '.json'), 'r') as json_file:
# #     state_series_dict = json.load(json_file)
# #     json_file.close()
#
# # group df_all by size
#
# # date = '2023_06_27'
# date = 'SimTrjs_RemoveAntsNearWall=False'
# # date = 'SimTrjs_RemoveAntsNearWall=True'
#
# df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
# df_all_dict = {k: v for k, v in df_gillespie.groupby('size')}
#
# time_series_dict, state_series_dict = Path.create_dicts(pd.concat(df_all_dict), ConfigSpace_SelectedStates,
#                                                         dictio_ts=None, dictio_ss=None)
# Path.save_dicts(time_series_dict, state_series_dict, name='_selected_states_' + date)
#
# Path.plot_paths(zip(['gillespie'], [df_all_dict]), time_series_dict, fig_name_add=date)

