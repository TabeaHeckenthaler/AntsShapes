from Analysis.PathPy.Path import *
from DataFrame.gillespie_dataFrame import df as df_all

add = '_gillespie'
# with open(os.path.join(network_dir, 'time_series_selected_states' + add + '.json'), 'r') as json_file:
#     time_series_dict = json.load(json_file)
#     json_file.close()
#
# with open(os.path.join(network_dir, 'state_series_selected_states' + add + '.json'), 'r') as json_file:
#     state_series_dict = json.load(json_file)
#     json_file.close()

# group df_all by size
df_all = df_all.groupby('size')
# put groups into dictionary
df_all_dict = {k: v for k, v in df_all}
# del df_all_dict['M']
# del df_all_dict['L']

time_series_dict, state_series_dict = Path.create_dicts(pd.concat(df_all_dict), ConfigSpace_SelectedStates,
                                                        dictio_ts=None, dictio_ss=None)
Path.save_dicts(time_series_dict, state_series_dict, name='_selected_states' + add)

Path.plot_paths(zip(['gillespie'], [df_all_dict]), time_series_dict)
