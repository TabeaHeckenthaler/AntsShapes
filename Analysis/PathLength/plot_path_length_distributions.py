from Analysis.PathLength.cutting_path_length import *
from Analysis.minimal_path_length.minimal_path_length import minimal_path_length_dict
from Analysis.PathLength.PathLength import penalized_path_length_dict, path_length_dict
from Analysis.average_carrier_number.averageCarrierNumber import averageCarrierNumber_dict


def plot_path_length_dist_cut_time(time_measure, path_length_measure, name='ps'):
    my_plot_class = Path_length_cut_off_df_ant()
    max_t_min = 27

    my_plot_class.cut_off_after_time(name, time_measure, path_length_measure, max_t=max_t_min*60)
    separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
                                                                  my_plot_class.plot_separately,
                                                                  'SPT',
                                                                  geometry=solver_geometry[my_plot_class.solver])

    fig, ax = plt.subplots(ncols=3, figsize=(15, 7))
    plt.show(block=False)

    my_plot_class.plot_means(separate_data_frames, None, None, ax[0])
    my_plot_class.plot_percent_of_solving(separate_data_frames, ax[1])
    my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, ax[2])

    ax[1].set_title(my_plot_class.solver + ',   ' + name + ',  max_t_min = ' + str(max_t_min) + ' min')
    save_fig(fig, name)


def plot_path_length_dist_cut_path(my_plot_class, path_length_measure, max_path, name):
    my_plot_class.cut_off_after_path_length(path_length_measure, max_path=max_path)
    separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver,
                                                                  my_plot_class.plot_separately,
                                                                  'SPT',
                                                                  geometry=solver_geometry[my_plot_class.solver])

    fig, ax = plt.subplots(ncols=3, figsize=(15, 7))
    plt.show(block=False)

    my_plot_class.plot_means(separate_data_frames, 'average Carrier Number', path_length_measure, ax[0])
    my_plot_class.plot_percent_of_solving(separate_data_frames, ax[1])
    my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, ax[2])

    ax[1].set_title(my_plot_class.solver + ',   ' + name + ',  max_path = ' + str(max_path) + ' * minimal p.length')
    save_fig(fig, name + '_' + str(max_path))


if __name__ == '__main__':
    # plot_means()

    # name = 'np_nt_cut_time'
    # # Problem:
    # # Solution:
    # # XL and L have similar path length, S and M don't not solve at all
    #
    # path_length_measure = 'path length/minimal path length[]'
    # time_measure = 'norm time [s]'
    # plot_path_length_dist_cut_time(time_measure, path_length_measure, name=name)

    # name = 'np_nst_cut_time'
    # # Problem: No data for S, because solving time is to short
    # # Solution:
    # # XL and L have similar path length, M doesn't not solve at all
    # path_length_measure = 'path length/minimal path length[]'
    # time_measure = 'norm solving time [s]'
    # plot_path_length_dist_cut_time(time_measure, path_length_measure, name=name)

    # name = 'npp_nt_cut_time'
    # # Problem: No data for S, because solving time is to short
    # # Solution:
    # # Only S(1), S(>1) and M seem to gather a lot of path length
    # # It seems that the larger the shape is, the less the shape gets stuck.
    # path_length_measure = 'penalized path length/minimal path length[]'
    # time_measure = 'norm time [s]'
    # plot_path_length_dist_cut_time(time_measure, path_length_measure, name=name)

    name = 'np_nt_cut_path'
    path_length_measure = 'path length/minimal path length[]'
    for max_path in range(10, 30, 3):
        my_plot_class = Path_length_cut_off_df_ant()

        my_plot_class.df['average Carrier Number'] = my_plot_class.df['filename'].map(averageCarrierNumber_dict)
        my_plot_class.df['minimal path length [length unit]'] = my_plot_class.df['filename'].map(minimal_path_length_dict)
        my_plot_class.df['path length [length unit]'] = my_plot_class.df['filename'].map(path_length_dict)
        my_plot_class.df['path length/minimal path length[]'] = my_plot_class.df['path length [length unit]']/ \
                                                                my_plot_class.df['minimal path length [length unit]']
        plot_path_length_dist_cut_path(my_plot_class, path_length_measure, max_path, name)

    name = 'npp_nt_cut_path'
    path_length_measure = 'penalized path length/minimal path length[]'
    for max_path in range(10, 30, 3):
        my_plot_class = Path_length_cut_off_df_ant()

        my_plot_class.df['average Carrier Number'] = my_plot_class.df['filename'].map(averageCarrierNumber_dict)
        my_plot_class.df['minimal path length [length unit]'] = my_plot_class.df['filename'].map(minimal_path_length_dict)
        my_plot_class.df['penalized path length [length unit]'] = my_plot_class.df['filename'].map(penalized_path_length_dict)
        my_plot_class.df['penalized path length/minimal path length[]'] = my_plot_class.df['penalized path length [length unit]']/ \
                                                                          my_plot_class.df['minimal path length [length unit]']
        plot_path_length_dist_cut_path(my_plot_class, path_length_measure, max_path, name)

