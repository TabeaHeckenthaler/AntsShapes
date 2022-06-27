from Analysis.Path_length_distributions import *


def plot_path_length_dist(time_measure, path_length_measure, name='ps'):
    # with open(name + '.pkl', 'rb') as file:
    #     separate_data_frames = pickle.load(file)
    my_plot_class = Path_length_cut_off_df_ant(time_measure=time_measure, path_length_measure=path_length_measure)
    fig, axs = my_plot_class.open_figure()
    my_plot_class.cut_off_after_time(time_measure, path_length_measure, max_t=30*60)
    separate_data_frames = my_plot_class.get_separate_data_frames('ant', my_plot_class.plot_separately, 'SPT',
                                                                  geometry=solver_geometry['ant'])
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(separate_data_frames, file)
    my_plot_class.plot_path_length_distributions(separate_data_frames, path_length_measure, axs, max_path=25)
    save_fig(fig, name + 'cut_of_time')


if __name__ == '__main__':
    path_length_measure = 'path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist(time_measure, path_length_measure, name='np_nt')

    path_length_measure = 'path length/minimal path length[]'
    time_measure = 'norm solving time [s]'
    plot_path_length_dist(time_measure, path_length_measure, name='np_nst')

    path_length_measure = 'penalized path length/minimal path length[]'
    time_measure = 'norm time [s]'
    plot_path_length_dist(time_measure, path_length_measure, name='npp_nt')
