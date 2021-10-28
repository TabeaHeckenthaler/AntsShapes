import pandas as pd
from os import listdir
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.trajectory import get, length_unit_func
from Analysis.PathLength import PathLength
from Setup.Maze import Maze
from Setup.Attempts import Attempts
from tqdm import tqdm

tqdm.pandas()


# TODO: add minimal path length!

def get_filenames(solver, size='', shape=''):
    if solver == 'human':
        shape_folder_naming = {'Large': 'large',
                               'Medium': 'medium',
                               'Small Near': 'small',
                               'Small Far': 'small2',
                               '': ''}
        return [filename for filename in listdir(SaverDirectories[solver])
                if ('_' in filename and shape_folder_naming[size] in filename)]
    else:
        return [filename for filename in listdir(SaverDirectories[solver])
                if size in filename and shape in filename]


columns = pd.Index(['filename', 'solver', 'size', 'maze size', 'shape', 'winner',
                    'communication', 'average Carrier Number', 'length unit',
                    'path length [length unit]', 'path length/minimal_path length[]',
                    'Attempts', 'path_length_during_attempts [length unit]'],
                   dtype='object')


class SingleExperiment(pd.DataFrame):
    def __init__(self, filename, solver, df: pd.DataFrame = None):
        if df is None:
            super().__init__([[filename, solver]], columns=['filename', 'solver'])
            self.add_information()
        else:
            super().__init__(df)

    def add_information(self):
        self['size'] = self['filename'].apply(lambda x: get(x).size, axis=1)
        self['shape'] = self['filename'].apply(lambda x: get(x).shape, axis=1)
        self['winner'] = self['filename'].apply(lambda x: get(x).winner, axis=1)
        self['communication'] = self['filename'].apply(lambda x: get(x).communication, axis=1)
        self['length unit'] = self[['solver']].apply(lambda x: length_unit_func(*x), axis=1)
        self['exit size [length unit]'] = self[['size', 'shape', 'solver']].apply(lambda x: Maze(*x).exit_size, axis=1)

        def maze_size(size: str):
            maze_s = {'Large': 'L',
                      'Medium': 'M',
                      'Small Far': 'S',
                      'Small Near': 'S'}
            if size in maze_s.keys():
                return maze_s[size]

        self['maze size'] = self[['size']].apply(lambda x: maze_size(*x), axis=1)

        self.applymap(lambda x: PathLength(x).per_experiment(), 'path length [length unit]')
        self.applymap(lambda x: PathLength(x).during_attempts(), 'path_length_during_attempts [length unit]')

        self['path length/exit size []'] = self.apply(
            lambda x: x['path length [length unit]'] / x['exit size [length unit]'], axis=1)
        self['average Carrier Number'] = self[['filename']].apply(
            lambda x: get(x).averageCarrierNumber(), axis=1)
        self['Attempts'] = self[['filename']].apply(
            lambda x: Attempts(get(x), 'extend'), axis=1)

        # self = self[list_of_columns]


class DataFrame(pd.DataFrame):
    def __init__(self, input, columns=None):
        if type(input) is pd.DataFrame:
            super().__init__(input, columns=columns)

        elif type(input) is list:
            super().__init__(pd.concat(input).reset_index(), columns=columns)

    def __add__(self, df_2):
        return DataFrame(pd.concat([self, df_2], ignore_index=True))

    def drop_non_existent(self):
        self.drop_duplicates(subset=['filename'], ignore_index=True)
        self.reset_index(inplace=True, drop=True)

        to_drop = []
        for solver in self.groupby(by='solver').groups.keys():
            filenames = get_filenames(solver)
            df_solver = self.iloc[self.groupby(by='solver').groups[solver]]
            for i in df_solver.index:
                if self.iloc[i]['filename'] not in filenames:
                    print('Dropped ' + str(self.iloc[i]['filename']))
                    to_drop.append(i)

        self.drop(index=to_drop, inplace=True)
        self.reset_index(drop=True, inplace=True)

    def save(self, name=df_dir + '.json'):
        # self.to_json(df_dir + ' - backup.json')
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

    def single_experiment(self, filename):
        df = self[(self['filename'] == filename)]
        return SingleExperiment(filename, df['solver'].values[0], df=df)


# human_couples = myDataFrame[(myDataFrame['average Carrier Number'] == 2) & (myDataFrame['solver'] == 'human')]


if __name__ == '__main__':
    myDataFrame = DataFrame(pd.read_json(df_dir + '.json'))
    myDataFrame.drop_non_existent()
    myDataFrame.save()

    # TODO: large_20201220135801_20201220140247, large_20201220142642_20201220143110 still in myDataFrame?

    myDataFrame['path length [length unit]'] = myDataFrame['filename'].progress_apply(
        lambda x: PathLength(get(x)).per_experiment())

    myDataFrame['path length/minimal_path length[]'] = myDataFrame.progress_apply(
        lambda x: x['path length [length unit]']/PathLength(get(x['filename'])).minimal(), axis=1)

    o = 1
    #
    # # TODO: Check in the data frame the path length of
    # #  'medium_20210901010920_20210901011020_20210901011020_20210901011022_20210901011022_20210901011433'
    # #
    # # myDataFrame = myDataFrame + myDataFrame.new_experiments(solver=solver)
    # # myDataFrame.save()
