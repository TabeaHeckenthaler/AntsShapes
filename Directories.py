from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'

home = path.abspath(__file__).split('\\')[0] + path.sep + path.join(*path.abspath(__file__).split(path.sep)[1:-1])
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)
work_dir = data_home + 'Pickled_Trajectories\\'
AntSaverDirectory = work_dir + 'Ant_Trajectories'
HumanSaverDirectory = work_dir + 'Human_Trajectories'
HumanHandSaverDirectory = work_dir + 'HumanHand_Trajectories'
DstarSaverDirectory = work_dir + 'Dstar_Trajectories'
SaverDirectories = {'ant': AntSaverDirectory,
                    'human': HumanSaverDirectory,
                    'humanhand': HumanHandSaverDirectory,
                    'dstar': DstarSaverDirectory}
PhaseSpaceDirectory = path.join(data_home, 'PhaseSpaces')


def ps_path(size, shape, solver, point_particle=False):
    if point_particle:
        return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '_pp.pkl')
    return path.join(PhaseSpaceDirectory, solver, size + '_' + shape + '.pkl')


def SetupDirectories():
    if not (path.isdir(AntSaverDirectory)):
        if not path.isdir('\\\\' + AntSaverDirectory.split('\\')[2]):
            return
        mkdir(AntSaverDirectory)
        mkdir(AntSaverDirectory + path.sep + 'OnceConnected')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion' + path.sep + 'OnceConnected')
    if not (path.isdir(HumanSaverDirectory)):
        mkdir(HumanSaverDirectory)
    if not (path.isdir(HumanHandSaverDirectory)):
        mkdir(HumanHandSaverDirectory)
    if not (path.isdir(DstarSaverDirectory)):
        mkdir(DstarSaverDirectory)
    return


SetupDirectories()
