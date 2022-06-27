from Analysis.Path_length_distributions import *


def plot_path_length_dist_cut_time(time_measure, path_length_measure, name='ps'):
    # with open(name + '.pkl', 'rb') as file:
    #     separate_data_frames = pickle.load(file)

    my_plot_class = Path_length_cut_off_df_ant(time_measure=time_measure, path_length_measure=path_length_measure)
    fig, axs = my_plot_class.open_figure()

    my_plot_class.cut_off_after_time(time_measure, path_length_measure, max_t=30*60)
    separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver, my_plot_class.plot_separately,
                                                                  'SPT',
                                                                  geometry=solver_geometry[my_plot_class.solver])

    with open(name + '.pkl', 'wb') as file:
        pickle.dump(separate_data_frames, file)

    my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs)
    my_plot_class.plot_means(axs)
    my_plot_class.plot_percent_of_solving(separate_data_frames)
    save_fig(fig, name)


def plot_path_length_dist_cut_path(time_measure, path_length_measure, name='ps'):
    # with open(name + '.pkl', 'rb') as file:
    #     separate_data_frames = pickle.load(file)

    my_plot_class = Path_length_cut_off_df_ant(time_measure=time_measure, path_length_measure=path_length_measure)
    fig, axs = my_plot_class.open_figure()

    my_plot_class.cut_off_after_path_length(path_length_measure=path_length_measure, max_path=25)
    separate_data_frames = my_plot_class.get_separate_data_frames(my_plot_class.solver, my_plot_class.plot_separately,
                                                                  'SPT', geometry=solver_geometry[my_plot_class.solver])

    with open(name + '.pkl', 'wb') as file:
        pickle.dump(separate_data_frames, file)

    my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs)
    my_plot_class.plot_means(axs)
    my_plot_class.plot_percent_of_solving(separate_data_frames)
    save_fig(fig, name)


if __name__ == '__main__':
    path_length_measure = 'path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist_cut_time(time_measure, path_length_measure, name='np_nt_cut_time')

    path_length_measure = 'path length/minimal path length[]'
    time_measure = 'norm solving time [s]'
    plot_path_length_dist_cut_time(time_measure, path_length_measure, name='np_nst_cut_time')

    path_length_measure = 'penalized path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist_cut_time(time_measure, path_length_measure, name='npp_nt_cut_time')

    path_length_measure = 'path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist_cut_path(time_measure, path_length_measure, name='np_nt_cut_path')

    path_length_measure = 'penalized path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist_cut_path(time_measure, path_length_measure, name='npp_nt_cut_path')