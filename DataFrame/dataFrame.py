import pandas as pd
from os import listdir
from Directories import SaverDirectories, df_dir
from trajectory_inheritance.get import get
from tqdm import tqdm
from copy import copy
import json
from DataFrame.SingleExperiment import SingleExperiment

pd.options.mode.chained_assignment = None
exp_solvers = ['ant', 'human', 'humanhand']


def get_filenames(solver, size='', shape='', free=False):
    if solver == 'human':
        shape_folder_naming = {'Large': 'large',
                               'Medium': 'medium',
                               'Small Near': 'small',
                               'Small Far': 'small2',
                               '': ''}
        return [filename for filename in listdir(SaverDirectories[solver])
                if ('_' in filename and shape_folder_naming[size] in filename)]
    if solver == 'ant':
        return [filename for filename in listdir(SaverDirectories[solver][free])
                if size in filename and shape in filename]
    if solver == 'humanhand':
        return [filename for filename in listdir(SaverDirectories[solver])]
    else:
        raise Exception('unknown solver: ' + solver)


columns = pd.Index(
    ['filename', 'solver', 'size', 'shape', 'winner', 'communication', 'length unit', 'initial condition',
     'force meter', 'fps', 'maze dimensions', 'load dimensions', 'comment', 'counted carrier number',
     'time [s]'],
    dtype='object')


class DataFrame(pd.DataFrame):
    def __init__(self, input, columns=None):
        if type(input) is pd.DataFrame:
            super().__init__(input, columns=columns)

        elif type(input) is list:
            super().__init__(pd.concat(input).reset_index(), columns=columns)

    def __add__(self, df_2):
        return DataFrame(pd.concat([self, df_2], ignore_index=True))

    @staticmethod
    def create():
        singleExperiments = []
        solver_filenames = {solver: get_filenames(solver) for solver in exp_solvers}

        for solver, filenames in solver_filenames.items():
            for filename in tqdm(filenames):
                singleExperiments.append(SingleExperiment(filename, solver))
        df = pd.concat(singleExperiments).reset_index(drop=True)

        DEBUG = 1
        # df.to_json(df_dir)

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
        to_load = set(get_filenames(solver, size=size, shape=shape, free=free)) \
                  - set(self['filename'].unique())
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
        pass
        # self['time [s]'] = self['filename'].progress_apply(lambda x: get(x).timer())
        # self['maze dimensions'], self['load dimensions'] = self['filename'].progress_apply(lambda x: get(x).geometry())

    def fill_column(self):
        for i, row in tqdm(self.iterrows()):
            if row['maze dimensions'] is None:
                geometry = get(row.filename).geometry()
                self.at[i, 'maze dimensions'] = geometry[0]
                self.at[i, 'load dimensions'] = geometry[1]

    def recalculate_experiment(self, filename):
        x = get(filename)
        df = self[self['filename'] == filename]
        if len(df) == 0:
            new_data_frame = self.copy()
        elif len(df) == 1:
            index = df.index[0]
            new_data_frame = self.drop([index]).reset_index(drop=True)
        else:
            raise ValueError('experiment double in df.')

        single = SingleExperiment(filename, x.solver)
        new_data_frame = DataFrame(new_data_frame) + DataFrame(pd.DataFrame(single))
        return new_data_frame

    def drop_experiment(self, filename):
        myDataFrame.drop(index=myDataFrame[myDataFrame['filename'] == filename].index, inplace=True)
        myDataFrame.reset_index(drop=True, inplace=True)


tqdm.pandas()
myDataFrame = DataFrame(pd.read_json(df_dir))
DEBUG = 1


def recalculating_cut_off_experiments():
    """
    These are the experiments which were cut off, because I used the wrong movie player.
    I retracked

    """
    done_filenames = ['XL_SPT_4630015_XLSpecialT_1_ants (part 1)',
                      'XL_SPT_4630018_XLSpecialT_1_ants',
                      'XL_SPT_4630019_XLSpecialT_1_ants (part 1)',
                      'XL_SPT_4640012_XLSpecialT_1_ants',
                      'XL_SPT_4640015_XLSpecialT_1_ants',
                      'XL_SPT_4640023_XLSpecialT_1_ants'
                      ]

    new_filenames = []

    for filename in new_filenames:
        myDataFrame = myDataFrame.recalculate_experiment(filename)

    with open('retracked.txt', 'w') as file:
        json.dump(done_filenames + new_filenames, file)


if __name__ == '__main__':
    # DataFrame.create()
    # TODO: add new contacts to contacts json file
    # TODO: Some of the human experiments don't have time [s].

    # drops = ['XL_SPT_4630015_XLSpecialT_1_ants', 'XL_SPT_4630019_XLSpecialT_1_ants']
    drops = ['S_H_4130039_smallH_1_ants']
    #
    for drop in drops:
        myDataFrame.drop_experiment(drop)

    # myDataFrame.add_column()
    for new_experiment in myDataFrame.new_experiments(solver='human', shape='SPT'):
        print(new_experiment['filename'].values[0])
        myDataFrame = myDataFrame + new_experiment

        ratio = new_experiment['path length [length unit]'].values[0] / \
                new_experiment['minimal path length [length unit]'].values[0]
        print(ratio)

        if ratio < 1.2 or ratio > 15:
            raise ValueError('weirdness in ' + new_experiment['filename'])

    myDataFrame.save()
