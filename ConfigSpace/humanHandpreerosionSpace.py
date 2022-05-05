from ConfigSpace import ConfigSpace_Maze
from trajectory_inheritance.exp_types import solver_geometry


def erode_centers():
    """for humanhand maze, which is not eroded correctly"""

    # 100, conf_space.space.shape[1] // 2, conf_space.space.shape[2] // 2
    one = [(100, 119, 311),
           (104, 109, 331), (108, 99, 351), (114, 89, 371),
           (104, 129, 291), (108, 139, 271), (114, 149, 251)]

    two = [(320, 112, -22), (320, 120, 22),
           (316, 129, 18), (312, 139, 36), (308, 149, 52),
           (316, 109, -20), (312, 99, -40), (308, 89, -60)]
    return two + one


def mask_around_centers(conf_space: ConfigSpace_Maze, centers: list):
    radiusx, radiusy, radiusz = 11, 11, 20
    mask = conf_space.empty_space()
    for center in centers:
        mask[center[0] - radiusx:center[0] + radiusx,
        center[1] - radiusy:center[1] + radiusy,
        center[2] - radiusz:center[2] + radiusz] = True
    return mask


if __name__ == '__main__':
    solver, shape = 'humanhand', 'SPT'
    geometry = solver_geometry[solver]

    for size in ['']:
        conf_space = ConfigSpace_Maze.ConfigSpace_Maze(solver, size, shape, geometry, name='')

        conf_space.load_space()
        conf_space.visualize_space(reduction=5)

        D = 1

        for centers in [erode_centers()]:
            mask = mask_around_centers(conf_space, centers)
            conf_space.visualize_space(space=mask, colormap='Oranges')
            import numpy as np
            space_to_delete = np.logical_and(conf_space.space, mask)
            conf_space.space = np.logical_and(conf_space.space, np.invert(space_to_delete))
            dir = '\\\\phys-guru-cs\\ants\\Tabea\\PyCharm_Data\\AntsShapes\\Configuration_Spaces\\SPT\\' \
                            '_SPT_MazeDimensions_humanhand_pre_erosion.pkl'
            conf_space.save_space(directory=dir)
            DEBUG = 2
