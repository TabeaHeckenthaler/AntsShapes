import pandas as pd
from os import listdir
from trajectory_inheritance.trajectory import solvers
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.trajectory import get, length_unit_func
from Analysis.Pathlength import path_length_per_experiment, path_length_during_attempts
from Setup.Maze import Maze
from Setup.Attempts import Attempts
from tqdm import tqdm


def get_filenames(solver, size='', shape=None):
    if solver == 'ant':
        if size is not None:
            pass
        if shape is not None:
            pass
        # TODO
        raise Exception('You still have to program this')
        # return [filename for filename in listdir(SaverDirectories[solver]) if 'ant' in filename
        #         if ('_' in filename and size in filename)]
    elif solver == 'human':
        shape_folder_naming = {'Large': 'large',
                               'Medium': 'medium',
                               'Small Near': 'small',
                               'Small Far': 'small2',
                               None: ''}
        return [filename for filename in listdir(SaverDirectories[solver])
                if ('_' in filename and shape_folder_naming[size] in filename)]
    else:
        return [filename for filename in listdir(SaverDirectories[solver])]


columns = pd.Index(['filename', 'solver', 'size', 'maze size', 'shape', 'winner',
                    'communication', 'average Carrier Number', 'length unit',
                    'path length [length unit]', 'exit size [length unit]',
                    'path length/exit size []', 'Attempts',
                    'path_length_during_attempts [length unit]'],
                   dtype='object')


class SingleExperiment(pd.DataFrame):

    def __init__(self, filename, solver):
        super().__init__([[filename, solver]], columns=['filename', 'solver'])
        self.add_information()

    def add_information(self):
        self['size'] = self[['filename', 'solver']].apply(lambda x: get(*x).size, axis=1)
        self['shape'] = self[['filename', 'solver']].apply(lambda x: get(*x).shape, axis=1)
        self['winner'] = self[['filename', 'solver']].apply(lambda x: get(*x).winner, axis=1)
        self['communication'] = self[['filename', 'solver']].apply(lambda x: get(*x).communication, axis=1)
        self['length unit'] = self[['solver']].apply(lambda x: length_unit_func(*x), axis=1)
        self['exit size [length unit]'] = self[['size', 'shape', 'solver']].apply(lambda x: Maze(*x).exit_size, axis=1)

        def maze_size(size: str):
            maze_s = {'Large': 'L',
                      'Medium': 'M',
                      'Small Far': 'S',
                      'Small Near': 'S'}
            if size in maze_s.keys():
                return maze_s[size]
            else:
                return size

        self['maze size'] = self[['size']].apply(lambda x: maze_size(*x), axis=1)

        self.apply_func(path_length_per_experiment, 'path length [length unit]')
        self.apply_func(path_length_during_attempts, 'path_length_during_attempts [length unit]')

        self['path length/exit size []'] = self.apply(
            lambda x: x['path length [length unit]'] / x['exit size [length unit]'], axis=1)
        self['average Carrier Number'] = self[['filename', 'solver']].apply(
            lambda x: get(*x).averageCarrierNumber(), axis=1)
        self['Attempts'] = self[['filename', 'solver']].apply(
            lambda x: Attempts(get(*x), 'extend'), axis=1)

        # self = self[list_of_columns]

    def apply_func(self, func: callable, column_name: str):
        # print('Calculating ' + column_name + ' with ' + func.__name__ + ' for ' + self.filename[0])
        self[column_name] = self[['filename', 'solver']].apply(lambda x: func(get(*x)), axis=1)


class DataFrame(pd.DataFrame):

    def __init__(self, input, columns=None):
        if type(input) is pd.DataFrame:
            super().__init__(input, columns=columns)

        elif type(input) is list:
            super().__init__(pd.concat(input).reset_index(), columns=columns)

    def __add__(self, df_2):
        return DataFrame(pd.concat([self, df_2], ignore_index=True))

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

    def save(self, name=df_dir + '.json'):
        # df.to_json(df_dir + ' - backup.json')
        self.to_json(name)

    def new_experiments(self, solver: str = 'ant', size: str = None):
        singleExperiments_list = []
        to_load = set(get_filenames(solver, size=size)) - set(self['filename'].unique())
        for filename in tqdm(to_load):
            print('Loading ' + filename + ' to df')
            singleExperiments_list.append(SingleExperiment(filename, solver))

        if len(singleExperiments_list) == 0:
            raise ValueError('You already loaded all experiments!')
        return DataFrame(singleExperiments_list, columns=columns)

# human_couples = myDataFrame[(myDataFrame['average Carrier Number'] == 2) & (myDataFrame['solver'] == 'human')]


if __name__ == '__main__':
    from Load_tracked_data.Load_Experiment import Load_Experiment, find_unpickled
    from trajectory_inheritance.trajectory import sizes
    solver = 'human'
    shape = 'SPT'
    for size in sizes[solver]:
        for filename in find_unpickled(solver, size, shape):
            x = Load_Experiment(solver, filename, [], True, 30, size=size, shape='SPT')
            x.play()
            x.save()

    myDataFrame = DataFrame(pd.read_json(df_dir + '.json'))

    # TODO: Check in the data frame the path length of
    #  'medium_20210901010920_20210901011020_20210901011020_20210901011022_20210901011022_20210901011433'

    myDataFrame = myDataFrame + myDataFrame.new_experiments(solver=solver)
    myDataFrame.save()
