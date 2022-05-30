from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.get import get
from Analysis.PathPy.Paths import plot_seperately
import json
from trajectory_inheritance.trajectory import solver_geometry
from Setup.Maze import Maze
from matplotlib import pyplot as plt


def flatten_dict(dict1):
    new = {}
    for key1, dict2 in dict1.items():
        for key2, value in dict2.items():
            new[str(key1) + '_' + str(key2)] = value
    return new


def av_distance_from_wall(distance: np.array, traj, config_space, exit_size):
    coords = np.stack([traj.position[:, 1], traj.position[:, 0], traj.angle]).transpose().tolist()
    inds = [config_space.coords_to_indices(*coord) for coord in coords]
    dist_integral = np.sum([distance[ind] for ind in inds])/len(inds)
    if type(config_space.indices_to_coords(dist_integral, 0, 0)[0]) is tuple or exit_size is tuple:
        DEBUG = 1
    dist_integral = config_space.indices_to_coords(dist_integral, 0, 0)[0]/exit_size
    return dist_integral


def calc_aver_distance():
    integral = {}

    for solver in solvers:
        geometry = solver_geometry[solver]
        ad = Altered_DataFrame()
        dfs = ad.get_separate_data_frames(solver, plot_seperately[solver], shape=shape)
        integral[solver] = {}

        if type(list(dfs.values())[0]) is dict:
            dfs = flatten_dict(dfs)

        for key, df in dfs.items():
            integral[solver][key] = {}
            print(key)
            d, cs, exit_size = None, None, None

            for filename in df['filename']:
                x = get(filename)
                if d is None or cs is None or exit_size is None:
                    cs = ConfigSpace_Maze(x.solver, x.size, shape, geometry)
                    cs.load_space()
                    d = cs.calculate_distance(cs.space, np.ones_like(cs.space))
                    exit_size = Maze(x).exit_size

                integral[solver][key][filename] = av_distance_from_wall(d, x, cs, exit_size)
                print(filename, integral[solver][key][filename])
    return integral


if __name__ == '__main__':
    shape = 'SPT'
    solvers = ['ant', 'human', 'humanhand']

    aver_distance = calc_aver_distance()

    with open('average_distance_to_boundary_scaled.json', 'w') as fp:
        json.dump(aver_distance, fp)
