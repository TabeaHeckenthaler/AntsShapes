from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import numpy as np
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.get import get
from Analysis.PathPy.Paths import plot_separately
import json
from trajectory_inheritance.trajectory import solver_geometry
from Setup.Maze import Maze
import os
from matplotlib import pyplot as plt
import pandas as pd
from DataFrame.plot_dataframe import save_fig


def flatten_dict(dict1, rep=3):  # TODO: better to make this recursive
    new = {}
    if rep == 3:
        for key1, dict2 in dict1.items():
            for key2, dict3 in dict2.items():
                for key3, value in dict3.items():
                    new[key3] = value

    if rep == 2:
        for key1, dict2 in dict1.items():
            for key2, value in dict2.items():
                new[key2] = value
    return new


def dic_of_df(dict1):
    new_df = pd.concat(flatten_list([v1.values() for v1 in [v for v in dict1.values()]]))
    return new_df


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class WallDistance(Altered_DataFrame):
    def __init__(self, solver, shape, geometry):
        super().__init__()
        self.solver = solver
        self.choose_experiments(solver=solver, shape=shape, geometry=geometry)
        self.wall_distances = self.get_results('av_distance_from_wall')
        # self.percents_sliding_along_wall = self.get_results('percent_sliding_along_wall')
        self.n_group_sizes = 5

    def plot_average_distance_from_wall(self, sep_df, axs, measure='av_distance_from_wall'):
        colors = ['green', 'red']
        maximum_value = self.get_maximum_value(sep_df, measure)

        bins = np.arange(0, maximum_value, maximum_value / 12)
        max_num_experiments = 1

        for i, (size, df_sizes) in enumerate(sep_df.items()):
            results = axs[i].hist([d[measure] for keys, d in df_sizes.items()], color=colors, bins=bins)
            axs[i].set_ylabel(size)
            max_num_experiments = max(np.max(results[0]), max_num_experiments)

        axs[-1].legend(sep_df[list(sep_df.keys())[-1]].keys())
        axs[-1].set_xlabel(measure)

        labelx = -0.05  # axes coords
        # for j in range(len(axs)):

        for ax in axs:
            ax.set_ylim([0, max_num_experiments + 1])

        # axs[-1].set_ylim([0, max_num_experiments + 1.5])
        axs[-1].yaxis.set_label_coords(labelx, 0.5)

    def get_results(self, measure: str) -> dict:
        if not os.path.exists(measure + '.json'):
            result = WallDistance.calc_for_all(measure=measure)
            with open(measure + '.json', 'w') as fp:
                json.dump(result, fp)

        else:
            with open(measure + '.json', 'r') as fp:
                result = json.load(fp)
        DEBUG = 1
        result = flatten_dict(result)
        calc = self.df.progress_apply(lambda exp: exp['filename'] in result.keys(), axis=1)

        if np.any(~calc):
            raise ValueError('You have to calculate ' + measure + ' for...')

        self.df[measure] = self.df.progress_apply(lambda exp: result[exp['filename']], axis=1)
        return result

    @staticmethod
    def get_maximum_value(dic, measure) -> float:
        df = pd.concat([v2[measure] for v2 in flatten_list([v1.values() for v1 in [v for v in dic.values()]])])
        return df.max()

    def add_new_experiments(self):
        pass

    @staticmethod
    def distance_from_wall(distance: np.array, traj, config_space, norm=1):
        coords = np.stack([traj.position[:, 1], traj.position[:, 0], traj.angle]).transpose().tolist()
        inds = [config_space.coords_to_indices(*coord) for coord in coords]
        d = [distance[ind] * norm for ind in inds]
        return d

    @staticmethod
    def av_distance_from_wall(distance: np.array, traj, config_space, exit_size):
        norm = config_space.indices_to_coords(1, 0, 0)[0]/exit_size
        dfw = WallDistance.distance_from_wall(distance, traj, config_space, norm=norm)
        dist_integral = np.sum(dfw)/len(dfw)
        # if type(config_space.indices_to_coords(dist_integral, 0, 0)[0]) is tuple or exit_size is tuple:
        #     DEBUG = 1
        return dist_integral

    @staticmethod
    def percent_sliding_along_wall(distance: np.array, traj, config_space, exit_size) -> float:
        norm = config_space.indices_to_coords(1, 0, 0)[0] / exit_size

        dfw_min = 0.04
        dfw = WallDistance.distance_from_wall(distance, traj, config_space, norm=norm)
        on_wall = [d < dfw_min for d in dfw]
        stuck = traj.stuck()

        sliding_along_wall = np.logical_and(np.array(on_wall[:-1]), ~np.array(stuck))
        perc_sliding_along_wall = np.sum(sliding_along_wall)/len(stuck)
        return perc_sliding_along_wall

    @staticmethod
    def calc_for_all(measure):
        results = {}

        for solver in solvers:
            geometry = solver_geometry[solver]
            ad = Altered_DataFrame()
            dfs = ad.get_separate_data_frames(solver, plot_separately[solver], shape=shape)
            results[solver] = {}
            d, cs, exit_size = None, None, None

            # if str(type(list(dfs.values())[0])) == "<class 'dict'>":
            df = dic_of_df(dfs)
            df = df.sort_values(by=['size'])

            for filename in df['filename']:
                x = get(filename)

                if d is None or cs is None or cs.size != x.size:
                    cs = ConfigSpace_Maze(x.solver, x.size, shape, geometry)
                    cs.load_space()
                    d = cs.calculate_distance(~cs.space, np.ones_like(cs.space))

                if exit_size is None:
                    exit_size = Maze(x).exit_size

                if measure == 'percent_sliding_along_wall':
                    results[solver][filename] = WallDistance.percent_sliding_along_wall(d, x, cs, exit_size)
                if measure == 'av_distance_from_wall':
                    results[solver][filename] = WallDistance.av_distance_from_wall(d, x, cs, exit_size)
                print(filename, results[solver][filename])
        return results

    def open_figure(self):
        fig, axs = plt.subplots(nrows=self.n_group_sizes, sharex=True)
        fig.subplots_adjust(hspace=0.2)
        plt.show(block=False)
        return fig, axs


if __name__ == '__main__':
    shape = 'SPT'
    solvers = ['ant', 'human', 'humanhand']
    geometries = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                  'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                  'humanhand': ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')}

    plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}

    for solver in ['ant']:
        wd = WallDistance(solver, shape, geometries[solver])
        sep_df = wd.get_separate_data_frames(solver, plot_separately[solver], shape, geometries[solver])

        fig, axs = wd.open_figure()
        wd.plot_average_distance_from_wall(sep_df, axs)
        save_fig(fig, 'wall_distance' + wd.solver)
