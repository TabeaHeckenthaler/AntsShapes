from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
# data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)
home = path.join(path.abspath(__file__).split('\\')[0] + path.sep, *path.abspath(__file__).split(path.sep)[1:-1])
data_home = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'PyCharm_Data', 'AntsShapes')

network_dir = path.join(data_home, 'Time_Series')

work_dir = path.join(data_home, 'Pickled_Trajectories')
SaverDirectories = {'ant': {True: path.join(work_dir, 'Ant_Trajectories', 'Free'),
                            False: path.join(work_dir, 'Ant_Trajectories', 'Slitted')},
                    'pheidole': path.join(work_dir, 'Pheidole_Trajectories'),
                    'human': path.join(work_dir, 'Human_Trajectories'),
                    'humanhand': path.join(work_dir, 'HumanHand_Trajectories'),
                    'gillespie': path.join(work_dir, 'Gillespie_Trajectories'),
                    'ps_simulation': path.join(work_dir, 'PS_simulation_Trajectories')}

mini_work_dir = path.join(data_home, 'mini_Pickled_Trajectories')
mini_SaverDirectories = {'ant': path.join(mini_work_dir, 'Ant_Trajectories'),
                         'pheidole': path.join(mini_work_dir, 'Pheidole_Trajectories'),
                         'human': path.join(mini_work_dir, 'Human_Trajectories'),
                         'humanhand': path.join(mini_work_dir, 'HumanHand_Trajectories'),
                         'gillespie': path.join(mini_work_dir, 'Gillespie_Trajectories'),
                         'ps_simulation': path.join(mini_work_dir, 'PS_simulation_Trajectories')}

# TODO: Rotation student: Change data_home to fit where you saved your space
PhaseSpaceDirectory = path.join(data_home, 'Configuration_Spaces')

excel_sheet_directory = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'Human Experiments')
contacts_dir = path.join(data_home, 'Contacts', 'ant')
df_dir = path.join(data_home, 'DataFrame', 'data_frame.xlsx')
df_sim_dir = path.join(data_home, 'DataFrame', 'data_frame_sim.json')
df_minimal_dir = path.join(data_home, 'DataFrame', 'data_frame_minimal.json')
maze_dimension_directory = path.join(home, 'Setup')

lists_exp_dir = path.join(data_home, 'DataFrame', 'excel_experiment_lists')

video_directory = path.join(home, 'Videos')
if not path.exists(video_directory):
    mkdir(video_directory)

original_movies_dir_ant = [
    '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Videos'.format(path.sep, path.sep, path.sep, path.sep, path.sep),
    '{0}{1}phys-guru-cs{2}ants{3}Lena{4}Movies'.format(path.sep, path.sep, path.sep, path.sep, path.sep),
    '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes'.format(path.sep, path.sep, path.sep, path.sep, path.sep)]
original_movies_dir_human = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Raw Data and Videos'.format(path.sep, path.sep, path.sep, path.sep, path.sep, path.sep)
original_movies_dir_humanhand = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Hand Experiments{5}Raw Data{6}' \
                                '2022_04_04 (Department Retreat)'.format(path.sep, path.sep, path.sep, path.sep,
                                                                         path.sep, path.sep, path.sep)

trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedPheidoleMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Pheidole Shapes Results'.format(path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep)
trackedHumanMovieDirectory = path.join(excel_sheet_directory, 'Output')
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'  # TODO

''' ANALYSIS '''
averageCarrierNumber_dir = path.join(home, 'Analysis', 'average_carrier_number', 'averageCarrierNumber.json')
minimal_path_length_dir = path.join(home, 'Analysis', 'minimal_path_length', 'minimal_path_length.json')
first_frame_dir = path.join(home, 'Analysis', 'first_frame.json')
path_length_dir = path.join(home, 'Analysis', 'Efficiency', 'path_length.json')
penalized_path_length_dir = path.join(home, 'Analysis', 'Efficiency', 'penalized_path_length.json')


#
# def SetupDirectories():
#     if not (path.isdir(SaverDirectories['ant'][False])):
#         if not path.isdir('\\\\' + SaverDirectories['ant'][False].split('\\')[2]):
#             return
#         mkdir(SaverDirectories['ant'])
#     if not (path.isdir(SaverDirectories['human'])):
#         mkdir(SaverDirectories['human'])
#     if not (path.isdir(SaverDirectories['humanhand'])):
#         mkdir(SaverDirectories['humanhand'])
#     if not (path.isdir(SaverDirectories['ps_simulation'])):
#         mkdir(SaverDirectories['ps_simulation'])
#
#     if not (path.isdir(mini_SaverDirectories['ant'])):
#         if not path.isdir('\\\\' + mini_SaverDirectories['ant'].split('\\')[2]):
#             return
#         mkdir(mini_SaverDirectories['ant'])
#     if not (path.isdir(mini_SaverDirectories['human'])):
#         mkdir(mini_SaverDirectories['human'])
#     if not (path.isdir(mini_SaverDirectories['humanhand'])):
#         mkdir(mini_SaverDirectories['humanhand'])
#     if not (path.isdir(mini_SaverDirectories['ps_simulation'])):
#         mkdir(mini_SaverDirectories['ps_simulation'])
#     return


def MatlabFolder(solver, size, shape, free=False):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not free:
            return path.join(trackedAntMovieDirectory, 'Slitted', shape_folder_naming[shape], size, 'Output Data')
        else:
            return path.join(trackedAntMovieDirectory, 'Free', 'Output Data', shape_folder_naming[shape])

    if solver == 'pheidole':
        return path.join(trackedPheidoleMovieDirectory, size, 'Output Data')

    if solver == 'human':
        return path.join(trackedHumanMovieDirectory, size, 'Data')

    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


def NewFileName(old_filename: str, solver: str, size: str, shape: str, expORsim: str) -> str:
    import glob
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        return size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            return filename.replace(old_filename.split('_')[0], size + '_' + shape)

        else:
            if solver in ['ant', 'pheidole']:
                if size + shape in filename or size + '_' + shape in filename:
                    return filename.replace(size + shape, size + '_' + shape)
                else:
                    raise ValueError('Your filename does not seem to be right.')
            elif solver in ['human', 'humanhand']:
                return filename

# SetupDirectories()
