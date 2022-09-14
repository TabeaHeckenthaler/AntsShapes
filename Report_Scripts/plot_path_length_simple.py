from Analysis.PathLength.cutting_path_length import *
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.PathLength.PathLength import penalized_path_length_dict, path_length_dict
from Analysis.average_carrier_number.averageCarrierNumber import averageCarrierNumber_dict


name = 'ant_np_nt_cut_path'
my_plot_class = Path_length_cut_off_df_ant()

my_plot_class.df['average Carrier Number'] = my_plot_class.df['filename'].map(averageCarrierNumber_dict)
my_plot_class.df['minimal path length [length unit]'] = my_plot_class.df['filename'].map(minimal_path_length_dict)
my_plot_class.df['path length [length unit]'] = my_plot_class.df['filename'].map(path_length_dict)
my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]'] / \
                                                        my_plot_class.df['minimal path length [length unit]']

max_path = 7
my_plot_class.cut_off_after_path_length('path length/minimal path length[]', max_path=max_path)
my_plot_class.plot_path_length_dist_cut_path('path length/minimal path length[]', max_path, name)
DEBUG = 1


name = 'human_np_nt_cut_path'
my_plot_class = Path_length_cut_off_df_human()

my_plot_class.df['average Carrier Number'] = my_plot_class.df['filename'].map(averageCarrierNumber_dict)
my_plot_class.df['minimal path length [length unit]'] = my_plot_class.df['filename'].map(minimal_path_length_dict)
my_plot_class.df['path length [length unit]'] = my_plot_class.df['filename'].map(path_length_dict)
my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]'] / \
                                                        my_plot_class.df['minimal path length [length unit]']
my_plot_class.plot_path_length_dist_cut_path('path length/minimal path length[]', None, name)
DEBUG = 1