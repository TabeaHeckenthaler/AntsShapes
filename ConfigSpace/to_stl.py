from ConfigSpace import ConfigSpace_Maze
from scipy.io import savemat


def add_connections(decimated):
    for i in [0, 1]:
        decimated[42+i, 17, 43] = True
        decimated[42+i, 16, 42] = True
        decimated[42+i, 15, 41] = True
        decimated[42+i, 15, 42] = True
        decimated[42+i, 14, 41] = True
        decimated[42+i, 15, 40] = True
        decimated[42+i, 14, 40] = True
        decimated[42+i, 13, 40] = True
    decimated[43, 16, 43] = True
    decimated[42, 16, 43] = True

    decimated[48, 15, 20] = True
    decimated[48, 15, 21] = True
    decimated[49, 15, 21] = True
    decimated[48, 16, 21] = True
    decimated[48, 16, 22] = True
    decimated[48, 16, 20] = True
    decimated[49, 16, 20] = True
    decimated[48, 17, 20] = True
    decimated[49, 17, 20] = True
    decimated[48, 18, 20] = True
    decimated[48, 18, 19] = True
    decimated[49, 18, 19] = True
    decimated[48, 16, 18] = False
    decimated[48, 17, 19] = False

    decimated[30, 30, 55] = True
    decimated[30, 30, 54] = True
    decimated[30, 30, 56] = True
    decimated[29, 30, 55] = True
    decimated[31, 30, 55] = True
    decimated[30, 30, 57] = True
    decimated[31, 30, 57] = True
    decimated[31, 30, 56] = True
    decimated[30, 31, 57] = True
    decimated[30, 31, 58] = True
    decimated[29, 30, 58] = True
    decimated[29, 31, 58] = True
    decimated[29, 30, 57] = True

    decimated[60, 30, 5] = True
    decimated[60:65, 28:31, 4:9] = True
    return decimated


def mirror(decimated):
    decimated[:, :, 62:] = decimated[:, :, :62][:, :, ::-1][:, ::-1]
    return decimated


solver, shape = 'human', 'SPT'
geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')

for size in ['Large']:
    conf_space = ConfigSpace_Maze.ConfigSpace_Maze(solver, size, shape, geometry, name='')

    conf_space.load_space()
    decimated = conf_space.space[::5, ::5, ::5]
    decimated = add_connections(decimated)
    decimated = mirror(decimated)
    conf_space.visualize_space(space=decimated)

    savemat('decimated.mat', {'dec_array': decimated})

    k = 1
