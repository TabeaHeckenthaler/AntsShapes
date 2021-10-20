import pandas as pd
from os import listdir
from trajectory_inheritance.trajectory import solvers
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.trajectory import get
from Analysis.Pathlength import path_length_per_experiment, path_length_during_attempts
from tqdm import tqdm
from Setup.Maze import Maze
from Setup.Attempts import Attempts
from Analysis.GeneralFunctions import flatten


def get_filenames(solver, size='_', shape=None):
    if solver == 'ant':
        if size is not None:
            pass
        if shape is not None:
            pass
        # TODO
        raise Exception('you still have to program this')
        # return [filename for filename in listdir(SaverDirectories[solver]) if 'ant' in filename
        #         if ('_' in filename and size in filename)]
    elif solver == 'human':
        shape_folder_naming = {'Large': 'large',
                               'Medium': 'medium',
                               'Small Near': 'small',
                               'Small Far': 'small2',
                               '_': '_'}
        return [filename for filename in listdir(SaverDirectories[solver])
                if ('_' in filename and shape_folder_naming[size] in filename)]
    else:
        return [filename for filename in listdir(SaverDirectories[solver])]


list_of_columns = ['filename', 'solver', 'size', 'maze size', 'shape',
                   'winner', 'communication', 'average Carrier Number',
                   'length unit', 'path length [length unit]', 'exit size [length unit]',
                   'path length/exit size []',
                   'Attempts', 'path_length_during_attempts [length unit]']


class MyDataFrame(pd.DataFrame):

    def drop_non_existent(self):
        self.drop_duplicates(subset=['filename'], inplace=True).reset_index()

        for solver in solvers:
            to_drop = []
            df_solver = self.iloc[self.groupby(by='solver').groups[solver]]
            for i in df_solver.index:
                # print(i)
                if self.iloc[i]['filename'] not in get_filenames(solver):
                    print('Dropped ' + str(self.iloc[i]['filename']))
                    to_drop.append(i)
            self.drop(index=to_drop, inplace=True)

        self.reset_index(drop=True, inplace=True).drop(columns=['index'], inplace=True)

    def apply_func(self, func, column_name):
        tqdm.pandas()
        print('Calculating ' + column_name + ' with ' + func.__name__)
        self[column_name] = self[['filename', 'solver']].progress_apply(lambda x: func(get(*x)), axis=1)

    def add_information(self):
        self['size'] = self[['filename', 'solver']].apply(lambda x: get(*x).size, axis=1)
        self['shape'] = self[['filename', 'solver']].apply(lambda x: get(*x).shape, axis=1)
        self['winner'] = self[['filename', 'solver']].apply(lambda x: get(*x).winner, axis=1)
        self['communication'] = self[['filename', 'solver']].apply(lambda x: x.communication, axis=1)
        # df['length unit'] = df[['solver']].apply(lambda x: length_unit_func(*x), axis=1) # TODO: install
        self['exit size [length unit]'] = self[['size', 'shape', 'solver']].apply(lambda x: Maze(*x).exit_size, axis=1)

        def maze_size(size):
            maze_s = {'Large': 'L',
                      'Medium': 'M',
                      'Small Far': 'S',
                      'Small Near': 'S'}
            if size in maze_s.keys():
                return maze_s[size]
            else:
                return size

        self['maze size'] = self[['size']].apply(lambda x: maze_size(*x), axis=1)

        self = self.apply_func(self, path_length_per_experiment, 'path length [length unit]')
        self = self.apply_func(self, path_length_during_attempts, 'path_length_during_attempts [length unit]')

        self['path length/exit size []'] = self.apply(
            lambda x: x['path length [length unit]'] / x['exit size [length unit]'], axis=1)
        self['average Carrier Number'] = self[['filename', 'solver']].progress_apply(
            lambda x: get(*x).averageCarrierNumber(), axis=1)
        self['Attempts'] = self[['filename', 'solver']].progress_apply(
            lambda x: Attempts(get(*x), 'extend'), axis=1)

        self = self[list_of_columns]

    def save_df(self):
        # df.to_json(df_dir + ' - backup.json')
        self.to_json(df_dir + '.json')

    def new_experiments_df(self, solver='ant'):
        new_experiment_dfs = pd.DataFrame()
        for filename in get_filenames(solver):
            if filename not in self['filename'].unique():
                new_experiment_df = pd.DataFrame([[filename, solver]],
                                                 columns=['filename', 'solver']) # TODO, not clear
                new_experiment_dfs = new_experiment_dfs.append(new_experiment_df.add_information(), ignore_index=True)
        return new_experiment_dfs


myDataFrame = MyDataFrame(pd.read_json(df_dir + '.json'))
