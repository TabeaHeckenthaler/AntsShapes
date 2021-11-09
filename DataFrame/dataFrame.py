import pandas as pd
from os import listdir
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.trajectory import get, length_unit_func
from Analysis.PathLength import PathLength
from Setup.Maze import Maze
from Setup.Attempts import Attempts
from tqdm import tqdm

tqdm.pandas()


def get_filenames(solver, size='', shape='', free=False):
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
                if size in filename and shape in filename and 'free' not in filename]


columns = pd.Index(['filename', 'solver', 'size', 'maze size', 'shape', 'winner',
                    'communication', 'average Carrier Number', 'length unit',
                    'path length [length unit]', 'path length/minimal_path length[]',
                    'Attempts', 'path_length_during_attempts [length unit]', 'initial condition'],
                   dtype='object')


class SingleExperiment(pd.DataFrame):
    def __init__(self, filename, solver, df: pd.DataFrame = None):
        if df is None:
            super().__init__([[filename, solver]], columns=['filename', 'solver'])
            self.add_information()
        else:
            super().__init__(df)

    def add_information(self):
        x = get(self['filename'][0])
        self['size'] = x.size
        self['shape'] = x.shape
        self['winner'] = x.winner
        self['communication'] = x.communication()
        self['length unit'] = length_unit_func(x.solver)
        self['exit size [length unit]'] = Maze(x).exit_size  # TODO: I don't think I need this anymore
        self['maze size'] = x.size[0]
        self['path length [length unit]'] = PathLength(x).per_experiment()
        self['path length/minimal_path length[]'] = self['path length [length unit]'] / PathLength(x).minimal()
        self['path_length_during_attempts [length unit]'] = PathLength(x).during_attempts()
        self['average Carrier Number'] = x.averageCarrierNumber()
        self['Attempts'] = Attempts(x, 'extend')
        self['initial condition'] = x.initial_cond()

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

    def new_experiments(self, solver: str = 'ant', size: str = '', shape: str = '', free=False):
        singleExperiments_list = []
        to_load = set(get_filenames(solver, size=size, shape=shape)) - set(self['filename'].unique())
        for filename in tqdm(to_load):
            print('Loading ' + filename + ' to df')
            singleExperiments_list.append(SingleExperiment(filename, solver))

        if len(singleExperiments_list) == 0:
            raise ValueError('You already loaded all experiments!')
        return DataFrame(singleExperiments_list, columns=columns)

    def single_experiment(self, filename):
        df = self[(self['filename'] == filename)]
        return SingleExperiment(filename, df['solver'].values[0], df=df)

    def sanity_check(self):
        problematic = self[(self['path length/minimal_path length[]'] < 1)
                           & (self['solver'] != 'ps_simulation')  # TODO: Still have to fix this
                           & (self['winner'] is True)]
        if len(problematic) > 0:
            raise ValueError('Your experiments \n' + str(problematic['filename']) + "\nare problematic")

    def add_column(self):
        pass
        # self['initial condition'] = self['filename'].progress_apply(lambda row: get(row).initial_cond())

# human_couples = myDataFrame[(myDataFrame['average Carrier Number'] == 2) & (myDataFrame['solver'] == 'human')]


if __name__ == '__main__':
    myDataFrame = DataFrame(pd.read_json(df_dir + '.json'))
    from DataFrame.plot_dataframe import how_many_experiments
    how_many_experiments(myDataFrame)

    # myDataFrame = myDataFrame + myDataFrame.new_experiments(solver='ant', shape='SPT')
    # myDataFrame.save()
