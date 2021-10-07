from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'

home = path.abspath(__file__).split('\\')[0] + path.sep + path.join(*path.abspath(__file__).split(path.sep)[1:-1])
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)

# work_dir = data_home + 'Pickled_Trajectories\\'
# AntSaverDirectory = work_dir + 'Ant_Trajectories'
# HumanSaverDirectory = work_dir + 'Human_Trajectories'
# HumanHandSaverDirectory = work_dir + 'HumanHand_Trajectories'
# PS_simulationSaverDirectory = work_dir + 'Dstar_Trajectories'
# PhaseSpaceDirectory = path.join(data_home, 'PhaseSpaces')
# SaverDirectories = {'ant': AntSaverDirectory,
#                     'human': HumanSaverDirectory,
#                     'humanhand': HumanHandSaverDirectory,
#                     'ps_simulation': PS_simulationSaverDirectory}

work_dir_new = data_home + 'Pickled_Trajectories_new\\'
AntSaverDirectory_new = work_dir_new + 'Ant_Trajectories'
HumanSaverDirectory_new = work_dir_new + 'Human_Trajectories'
HumanHandSaverDirectory_new = work_dir_new + 'HumanHand_Trajectories'
PS_simulationSaverDirectory_new = work_dir_new + 'PS_simulation_Trajectories'
GillespieSaverDirectory_new = work_dir_new + 'Gillespie_Trajectories'

SaverDirectories_new = {'ant': AntSaverDirectory_new,
                        'human': HumanSaverDirectory_new,
                        'humanhand': HumanHandSaverDirectory_new,
                        'gillespie': GillespieSaverDirectory_new,
                        'ps_simulation': PS_simulationSaverDirectory_new}

PhaseSpaceDirectory_new = path.join(data_home, 'PhaseSpaces')


# def ps_path(size, shape, solver, point_particle=False):
#     if point_particle:
#         return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '_pp.pkl')
#     return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '.pkl')
#
#
# def SetupDirectories():
#     if not (path.isdir(AntSaverDirectory)):
#         if not path.isdir('\\\\' + AntSaverDirectory.split('\\')[2]):
#             return
#         mkdir(AntSaverDirectory)
#         mkdir(AntSaverDirectory + path.sep + 'OnceConnected')
#         mkdir(AntSaverDirectory + path.sep + 'Free_Motion')
#         mkdir(AntSaverDirectory + path.sep + 'Free_Motion' + path.sep + 'OnceConnected')
#     if not (path.isdir(HumanSaverDirectory)):
#         mkdir(HumanSaverDirectory)
#     if not (path.isdir(HumanHandSaverDirectory)):
#         mkdir(HumanHandSaverDirectory)
#     if not (path.isdir(PS_simulationSaverDirectory)):
#         mkdir(PS_simulationSaverDirectory)
#     return


def SetupDirectories_new():
    if not (path.isdir(AntSaverDirectory_new)):
        if not path.isdir('\\\\' + AntSaverDirectory_new.split('\\')[2]):
            return
        mkdir(AntSaverDirectory_new)
    if not (path.isdir(HumanSaverDirectory_new)):
        mkdir(HumanSaverDirectory_new)
    if not (path.isdir(HumanHandSaverDirectory_new)):
        mkdir(HumanHandSaverDirectory_new)
    if not (path.isdir(PS_simulationSaverDirectory_new)):
        mkdir(PS_simulationSaverDirectory_new)
    return


SetupDirectories_new()  # TODO: Delete
# SetupDirectories()