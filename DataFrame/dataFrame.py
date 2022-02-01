import pandas as pd
from os import listdir
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.trajectory import get, length_unit_func
from Analysis.PathLength import PathLength
from Setup.Attempts import Attempts
from tqdm import tqdm
from copy import copy

pd.options.mode.chained_assignment = None


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
                    'communication', 'length unit', 'average Carrier Number', 'Attempts',
                    'path length during attempts [length unit]', 'path length [length unit]', 'initial condition',
                    'minimal path length [length unit]', 'force meter', 'fps', 'maze dimensions', 'load dimensions',
                    'comment', 'counted carrier number'],
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
        self['size'] = str(x.size)
        self['shape'] = str(x.shape)
        self['winner'] = bool(x.winner)
        self['fps'] = int(x.fps)
        self['communication'] = bool(x.communication())
        self['length unit'] = str(length_unit_func(x.solver))
        self['maze size'] = str(x.size[0])
        self['path length [length unit]'] = float(PathLength(x).per_experiment())

        if x.shape != 'SPT':
            self['path length during attempts [length unit]'] = PathLength(x).during_attempts()

        self['minimal path length [length unit]'] = PathLength(x).minimal()
        self['average Carrier Number'] = float(x.averageCarrierNumber())
        self['Attempts'] = Attempts(x, 'extend')
        self['initial condition'] = str(x.initial_cond())
        self['force meter'] = bool(x.has_forcemeter())
        self['maze dimensions'], self['load dimensions'] = x.geometry()
        self['counted carrier number'] = None

        # self = self[list_of_columns]


class DataFrame(pd.DataFrame):
    def __init__(self, input, columns=None):
        if type(input) is pd.DataFrame:
            super().__init__(input, columns=columns)

        elif type(input) is list:
            super().__init__(pd.concat(input).reset_index(), columns=columns)

    def __add__(self, df_2):
        return DataFrame(pd.concat([self, df_2], ignore_index=True))

    def clone(self):
        return copy(self)

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

    def save(self, name=df_dir):
        self.to_json(name)

    def back_up(self):
        if bool(int(input('Check the DataFrame carefully!'))):
            self.to_json(df_dir + ' - backup.json')

    def new_experiments(self, solver: str = 'ant', size: str = '', shape: str = '', free=False):
        to_load = set(get_filenames(solver, size=size, shape=shape)) - set(self['filename'].unique())
        for filename in tqdm(to_load):
            print('Loading ' + filename + ' to df')
            yield SingleExperiment(filename, solver)

    def single_experiment(self, filename):
        df = self[(self['filename'] == filename)]
        return SingleExperiment(filename, df['solver'].values[0], df=df)

    def sanity_check(self):
        problematic = self[(self['path length/minimal path length[]'] < 1)
                           & (self['solver'] != 'ps_simulation')  # TODO: Still have to fix this
                           & (self['winner'] is True)]
        if len(problematic) > 0:
            raise ValueError('Your experiments \n' + str(problematic['filename']) + "\nare problematic")

    def add_column(self):
        # with open('participant_count_dict.txt', 'r') as json_file:
        #     participant_count_dict = json.load(json_file)
        #
        # for i, exp in self.iterrows():
        #     if exp['filename'] in participant_count_dict.keys():
        #         self.at[i, 'average Carrier Number'] = participant_count_dict[exp['filename']]

        self['maze dimensions'], self['load dimensions'] = self['filename'].progress_apply(lambda x: get(x).geometry())

    def fill_column(self):
        for i, row in tqdm(self.iterrows()):
            if row['maze dimensions'] is None:
                geometry = get(row.filename).geometry()
                self.at[i, 'maze dimensions'] = geometry[0]
                self.at[i, 'load dimensions'] = geometry[1]

    def recalculate_experiment(self, filename):
        x = get(filename)
        index = myDataFrame[myDataFrame['filename'] == filename].index[0]
        new_data_frame = DataFrame(self.drop([index]).reset_index(drop=True))

        single = SingleExperiment(filename, x.solver)
        new_data_frame = new_data_frame + single
        return new_data_frame

tqdm.pandas()
myDataFrame = DataFrame(pd.read_json(df_dir))

filename = 'M_SPT_4700022_MSpecialT_1_ants'
new_data_frame = myDataFrame.recalculate_experiment(filename)
DEBUG = 1

if __name__ == '__main__':
    # TODO: add new contacts to contacts json file
    # from DataFrame.plot_dataframe import how_many_experiments
    # how_many_experiments(myDataFrame)

    for new_experiment in myDataFrame.new_experiments(solver='ant', shape='SPT'):
        print(new_experiment['filename'])
        myDataFrame = myDataFrame + new_experiment
        myDataFrame.save()

    # TODO
    # for i in [114, 120]:
    ## L_SPT_4080033_SpecialT_1_ants (part 1)
    ## L_SPT_4090010_SpecialT_1_ants (part 1)
    #     myDataFrame.at[i, 'initial condition'] = 'front'
    #     myDataFrame.at[i, 'comment'] = 'Here the beginning of the movie is cut out. ' \
    #                                    'I suspect it started in front, not the back. ' \
    #                                    'Maybe, I should add a smooth connector.'
