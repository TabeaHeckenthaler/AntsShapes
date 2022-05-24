from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from matplotlib import pyplot as plt
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.get import get
from Analysis.PathPy.Paths import plot_seperately
import json
from trajectory_inheritance.trajectory import solver_geometry


def flatten_dict(dict1):
    new = {}
    for key1, dict2 in dict1.items():
        for key2, value in dict2.items():
            new[str(key1) + '_' + str(key2)] = value
    return new


def wall_closeness(distance: np.array, traj, config_space):
    coords = np.stack([x.position[:, 1], x.position[:, 0], x.angle]).transpose().tolist()
    inds = [config_space.coords_to_indices(*coord) for coord in coords]
    dist_integral = np.sum([distance[ind] for ind in inds])/len(inds)
    return dist_integral


if __name__ == '__main__':
    shape = 'SPT'
    solvers = ['ant', 'human', 'humanhand']
    integral = {}

    for solver in solvers:
        geometry = solver_geometry[solver]
        ad = Altered_DataFrame()
        dfs = ad.get_separate_data_frames(solver, plot_seperately[solver], shape=shape)
        integral[solver] = {}

        if type(list(dfs.values())[0]) is dict:
            dfs = flatten_dict(dfs)

        for key, df in dfs.items():
            print(key)
            d, cs = None, None

            for filename in df['filename']:
                x = get(filename)

                if d is None or cs is None:
                    cs = ConfigSpace_Maze(x.solver, x.size, shape, geometry)
                    cs.load_space()
                    d = cs.calculate_distance(cs.space, np.ones_like(cs.space))

                integral[solver][filename] = wall_closeness(d, x, cs)
                print(filename, integral[solver][filename])
                # plt.imshow(distance[200])
                # ps.visualize_space(reduction=4)

    with open('sliding_behaviour.json', 'w') as fp:
        json.dump(integral, fp)
