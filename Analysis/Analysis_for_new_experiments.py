from DataFrame.dataFrame import DataFrame, pd, df_dir
from Analysis.Efficiency.PathLength import PathLength
from Analysis.PathPy.Path import Path
from Analysis.find_first_starting_frame import FirstFrame

# TODO: you still have to update average Carrier Number dictionary

# trajectory filenames to recalculate
to_pop = []

#  DataFrame
myDataFrame = DataFrame(pd.read_json(df_dir))
for new_experiment in myDataFrame.new_experiments(solver='ant', shape='SPT'):
    print(new_experiment['filename'].values[0])
    myDataFrame = myDataFrame + new_experiment
myDataFrame.save()


#  Path length
path_length_dict, penalized_path_length_dict = PathLength.get_dicts()
for pop in to_pop:
    print(path_length_dict.pop(pop))
    print(penalized_path_length_dict.pop(pop))
path_length_dict, penalized_path_length_dict = PathLength.add_to_dict(myDataFrame, path_length_dict,
                                                                      penalized_path_length_dict)
PathLength.save_dicts(path_length_dict, penalized_path_length_dict)


#  States
time_series_dict, state_series_dict = Path.get_dicts()
for pop in to_pop:
    print(time_series_dict.pop(pop))
    print(state_series_dict.pop(pop))
raise ValueError
time_series_dict, state_series_dict = Path.add_to_dict(myDataFrame, time_series_dict, state_series_dict)
Path.save_dicts(time_series_dict, state_series_dict)


# First Frame
first_frame_dict = FirstFrame.get_dict()
for pop in to_pop:
    print(first_frame_dict.pop(pop))
first_frame_dict = FirstFrame.add_to_dict(myDataFrame, first_frame_dict)
FirstFrame.save_dict(first_frame_dict)

