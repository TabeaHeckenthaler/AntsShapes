from os import getcwd, path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'

home = __file__.split('\\')[0] + '\\' + path.join(*__file__.split('\\')[1:-1])
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


def SetupDirectories():
    if not (path.isdir(AntSaverDirectory)):
        breakpoint()
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