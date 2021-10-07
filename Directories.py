from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
home = path.abspath(__file__).split('\\')[0] + path.sep + path.join(*path.abspath(__file__).split(path.sep)[1:-1])
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)

work_dir_new = data_home + 'Pickled_Trajectories_new\\'

SaverDirectories_new = {'ant': work_dir_new + 'Ant_Trajectories',
                        'human': work_dir_new + 'Human_Trajectories',
                        'humanhand': work_dir_new + 'HumanHand_Trajectories',
                        'gillespie': work_dir_new + 'Gillespie_Trajectories',
                        'ps_simulation': work_dir_new + 'PS_simulation_Trajectories'}

PhaseSpaceDirectory_new = path.join(data_home, 'PhaseSpaces')


def ps_path(size, shape, solver, point_particle=False):
    if point_particle:
        return path.join(PhaseSpaceDirectory_new, solver, size + '_' + shape + '_pp.pkl')
    return path.join(PhaseSpaceDirectory_new, solver, size + '_' + shape + '.pkl')


def SetupDirectories():
    if not (path.isdir(SaverDirectories_new['ant'])):
        if not path.isdir('\\\\' + SaverDirectories_new['ant'].split('\\')[2]):
            return
        mkdir(SaverDirectories_new['ant'])
    if not (path.isdir(SaverDirectories_new['human'])):
        mkdir(SaverDirectories_new['human'])
    if not (path.isdir(SaverDirectories_new['humanhand'])):
        mkdir(SaverDirectories_new['humanhand'])
    if not (path.isdir(SaverDirectories_new['ps_simulation'])):
        mkdir(SaverDirectories_new['ps_simulation'])
    return


trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'


def MatlabFolder(solver, size, shape, free):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not free:
            return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
                shape] + path.sep + size + path.sep + 'Output Data'
        if free:
            return trackedAntMovieDirectory + path.sep + 'Free' + path.sep + 'Output Data' + path.sep + \
                   shape_folder_naming[shape]
    if solver == 'human':
        if not free:
            return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'
        if free:
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
