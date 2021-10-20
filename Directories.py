from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
home = path.abspath(__file__).split('\\')[0] + path.sep + path.join(*path.abspath(__file__).split(path.sep)[1:-1])
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)

work_dir = data_home + 'Pickled_Trajectories\\'

SaverDirectories = {'ant': work_dir + 'Ant_Trajectories',
                    'human': work_dir + 'Human_Trajectories',
                    'humanhand': work_dir + 'HumanHand_Trajectories',
                    'gillespie': work_dir + 'Gillespie_Trajectories',
                    'ps_simulation': work_dir + 'PS_simulation_Trajectories'}

PhaseSpaceDirectory = path.join(data_home, 'PhaseSpaces')

excel_sheet_directory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments'.format(path.sep, path.sep,
                                                                                       path.sep,
                                                                                       path.sep, path.sep)

df_dir = data_home + 'DataFrame\\data_frame'


def ps_path(size, shape, solver, point_particle=False):
    if point_particle:
        return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '_pp.pkl')
    return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '.pkl')


def SetupDirectories():
    if not (path.isdir(SaverDirectories['ant'])):
        if not path.isdir('\\\\' + SaverDirectories['ant'].split('\\')[2]):
            return
        mkdir(SaverDirectories['ant'])
    if not (path.isdir(SaverDirectories['human'])):
        mkdir(SaverDirectories['human'])
    if not (path.isdir(SaverDirectories['humanhand'])):
        mkdir(SaverDirectories['humanhand'])
    if not (path.isdir(SaverDirectories['ps_simulation'])):
        mkdir(SaverDirectories['ps_simulation'])
    return


trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'


def MatlabFolder(solver, size, shape):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
            shape] + path.sep + size + path.sep + 'Output Data'

    if solver == 'human':
        return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'

    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


def NewFileName(old_filename, size, shape, expORsim):
    import glob
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        # findall(r'[\d.]+', 'TXL1_sim_255')[1] #this is a function able to read the last digit of the string
        filename = size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            filename = filename.replace(old_filename.split('_')[0], size + '_' + shape)
        else:
            filename = filename.replace(size + shape, size + '_' + shape)
    return filename


SetupDirectories()
