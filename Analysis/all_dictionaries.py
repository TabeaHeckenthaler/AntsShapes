from DataFrame.dataFrame import myDataFrame

# average carrier number
from Analysis.average_carrier_number.averageCarrierNumber import extend_dictionary
extend_dictionary(myDataFrame)

# path length
from Analysis.Efficiency.PathLength import PathLength, path_length_dir
path_length_dict = PathLength.get_dict(path_length_dir)
path_length_dict = PathLength.add_to_dict(myDataFrame, path_length_dict)
PathLength.save_dict(path_length_dict, path_length_dir)

# minimal path length
from Analysis.minimal_path_length.minimal_path_length import MinimalDataFrame
myMinimalDataFrame = MinimalDataFrame()
minimal_path_length_dict = MinimalDataFrame.get_dict()
minimal_path_length_dict = myMinimalDataFrame.update_dict(myDataFrame, minimal_path_length_dict)
myMinimalDataFrame.save_dict(minimal_path_length_dict)

# penalized path length
from Analysis.Efficiency.PenalizedPathLength import PenalizedPathLength, penalized_path_length_dir
penalized_path_length_dict = PenalizedPathLength.get_dict(penalized_path_length_dir)
penalized_path_length_dict = PenalizedPathLength.add_to_dict(myDataFrame, penalized_path_length_dict)
print('I think I overwrote my Path length dict')
PenalizedPathLength.save_dict(penalized_path_length_dict, penalized_path_length_dir)

# states
from Analysis.PathPy.Path import Path, ConfigSpace_SelectedStates
time_series_dict, state_series_dict = Path.get_dicts('_selected_states')
ConfigSpace_class = ConfigSpace_SelectedStates
to_add = Path.find_missing(myDataFrame)
print(to_add)
time_series_dict_selected_states, state_series_dict_selected_states = Path.add_to_dict(to_add,
                                                                                       ConfigSpace_class,
                                                                                       time_series_dict,
                                                                                       state_series_dict)

from Analysis.Path_Length_States import create_bar_chart
# from matplotlib import pyplot as plt
# filenames = to_add['filename'].tolist()
# df = myDataFrame.copy()
# df['time series'] = df['filename'].map(time_series_dict)
# df = df[df['filename'].isin(filenames)]
# fig10, ax10 = plt.subplots()
# plot_bar_chart(df, ax10)

Path.save_dicts(time_series_dict_selected_states, state_series_dict_selected_states, name='_selected_states')
