from Analysis.PathPy.Paths import PathsTimeStamped, plot_separately
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.exp_types import exp_types, solver_geometry


def split_up(solver):
    shape = 'SPT'

    all_paths = PathsTimeStamped(solver, shape, solver_geometry[solver], '')
    for size in exp_types[shape][solver]:
        paths_of_size = PathsTimeStamped(solver, shape, solver_geometry[solver], size)
        paths_of_size.load_paths()
        paths_of_size.load_time_stamped_paths()

        all_paths.time_series.update(paths_of_size.time_series)
        all_paths.time_stamped_series.update(paths_of_size.time_stamped_series)

    df = Altered_DataFrame()
    dfs = df.get_separate_data_frames(solver, plot_separately[solver], shape='SPT', geometry=solver_geometry[solver],
                                      initial_cond='back')

    for key_group_size, ds in dfs.items():
        for key_communication, experiments_df in ds.items():
            filenames = list(experiments_df['filename'])
            paths = PathsTimeStamped(solver, shape, solver_geometry[solver])

            paths.time_series = {filename: all_paths.time_series[filename] for filename in filenames}
            paths.time_stamped_series = {filename: all_paths.time_stamped_series[filename] for filename in filenames}

            name = 'paths_ants_' + key_group_size.replace('>', 'more_than') + '_' + key_communication
            paths.save_paths(paths.save_dir(name=name))
            paths.save_timestamped_paths(paths.save_dir_timestamped(name=name))
            paths.save_csv(paths.save_dir_timestamped(name=name))


def humans_split_up():
    solver, shape, geometry = 'human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

    all_paths = PathsTimeStamped(solver, shape, geometry, '')
    for size in exp_types[shape][solver]:
        paths_of_size = PathsTimeStamped(solver, shape, geometry, size)
        paths_of_size.load_paths()
        paths_of_size.load_time_stamped_paths()

        all_paths.time_series.update(paths_of_size.time_series)
        all_paths.time_stamped_series.update(paths_of_size.time_stamped_series)

    df = Altered_DataFrame()
    dfs = df.get_separate_data_frames(solver, plot_separately[solver], shape='SPT', geometry=solver_geometry[solver],
                                      initial_cond='back')

    for key_group_size, ds in dfs.items():
        for key_communication, experiments_df in ds.items():
            filenames = list(experiments_df['filename'])
            paths = PathsTimeStamped(solver, shape, geometry)

            paths.time_series = {filename: all_paths.time_series[filename] for filename in filenames}
            paths.time_stamped_series = {filename: all_paths.time_stamped_series[filename] for filename in filenames}

            name = 'paths_humans_' + key_group_size.replace('>', 'more_than') + '_' + key_communication
            paths.save_paths(paths.save_dir(name=name))
            paths.save_timestamped_paths(paths.save_dir_timestamped(name=name))
            paths.save_csv(paths.save_dir_timestamped(name=name))

