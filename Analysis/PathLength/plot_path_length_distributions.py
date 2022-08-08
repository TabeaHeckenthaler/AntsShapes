from Analysis.PathLength.cutting_path_length import *
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.PathLength.PathLength import penalized_path_length_dict, path_length_dict
from Analysis.average_carrier_number.averageCarrierNumber import averageCarrierNumber_dict


def setup():
    my_plot_class.df['average Carrier Number'] = my_plot_class.df['filename'].map(averageCarrierNumber_dict)
    my_plot_class.df['minimal path length [length unit]'] = my_plot_class.df['filename'].map(minimal_path_length_dict)
    my_plot_class.df['path length [length unit]'] = my_plot_class.df['filename'].map(path_length_dict)
    my_plot_class.df['penalized path length [length unit]'] = my_plot_class.df['filename'].map(penalized_path_length_dict)


if __name__ == '__main__':

    ##################
    # ANTS
    ##################
    name = 'ant_np_nt_cut_path'
    path_length_measure = 'path length/minimal path length[]'
    for max_path in range(10, 30, 3):
        my_plot_class = Path_length_cut_off_df_ant()

        setup()
        my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]']/ \
                                                                my_plot_class.df['minimal path length [length unit]']
        my_plot_class.cut_off_after_path_length(path_length_measure, max_path=max_path)
        my_plot_class.plot_path_length_dist_cut_path(path_length_measure, max_path, name)

    name = 'ant_npp_nt_cut_path'
    path_length_measure = 'penalized path length/minimal path length[]'
    for max_path in range(10, 30, 3):
        my_plot_class = Path_length_cut_off_df_ant()

        setup()
        my_plot_class.df['penalized path length/minimal path length[]'] = my_plot_class.df['penalized path length [length unit]']/ \
                                                                          my_plot_class.df['minimal path length [length unit]']
        my_plot_class.cut_off_after_path_length(path_length_measure, max_path=max_path)
        my_plot_class.plot_path_length_dist_cut_path(path_length_measure, max_path, name)

    # ##################
    # # HUMANS
    # ##################
    name = 'human_np_nt_cut_path'
    path_length_measure = 'path length/minimal path length[]'
    my_plot_class = Path_length_cut_off_df_human()
    setup()
    my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]']/ \
                                                            my_plot_class.df['minimal path length [length unit]']
    my_plot_class.plot_path_length_dist_cut_path(path_length_measure, None, name)

    name = 'human_npp_nt_cut_path'
    path_length_measure = 'penalized path length/minimal path length[]'
    my_plot_class = Path_length_cut_off_df_humanhand()
    setup()
    my_plot_class.df['penalized path length/minimal path length[]'] = my_plot_class.df['penalized path length [length unit]']/ \
                                                                      my_plot_class.df['minimal path length [length unit]']
    my_plot_class.cut_off_after_path_length(path_length_measure)
    my_plot_class.plot_path_length_dist_cut_path(path_length_measure, None, name)

    ##################
    # HUMANHAND
    ##################
    name = 'humanhand_np_nt_cut_path'
    path_length_measure = 'path length/minimal path length[]'
    my_plot_class = Path_length_cut_off_df_humanhand()
    setup()
    my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]']/ \
                                                            my_plot_class.df['minimal path length [length unit]']
    my_plot_class.plot_path_length_dist_cut_path(path_length_measure, None, name)

    name = 'humanhand_npp_nt_cut_path'
    path_length_measure = 'penalized path length/minimal path length[]'
    my_plot_class = Path_length_cut_off_df_humanhand()
    setup()
    my_plot_class.df['penalized path length/minimal path length[]'] = my_plot_class.df['penalized path length [length unit]']/ \
                                                                      my_plot_class.df['minimal path length [length unit]']
    my_plot_class.cut_off_after_path_length(path_length_measure)
    my_plot_class.plot_path_length_dist_cut_path(path_length_measure, None, name)

