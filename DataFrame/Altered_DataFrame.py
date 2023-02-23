from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_humanhand import ExcelSheet
from trajectory_inheritance.exp_types import solver_geometry
from Directories import df_dir, averageCarrierNumber_dir, lists_exp_dir, original_movies_dir_ant, \
    original_movies_dir_human, original_movies_dir_humanhand
import json
import os
import numpy as np

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}, 'pheidole': {'': []}}


class Altered_DataFrame:
    def __init__(self, df=None):
        if df is None:
            self.df = myDataFrame.clone()
        else:
            self.df = df

    def choose_experiments(self, solver=None, shape=None, geometry=None, size=None, communication=None,
                           init_cond: str = 'back', winner: bool = None, free: bool = None):
        """
        Get trajectories based on the trajectories saved in myDataFrame.
        :param geometry: geometry of the maze, given by the names of the excel files with the dimensions
        :param number: How many trajectories do you want to have in your list?
        :param solver: str
        :param size: str
        :param shape: str
        :return: list of objects, that are of the class or subclass Trajectory
        """
        df = self.df
        if 'shape' in df.columns and shape is not None:
            df = df[(df['shape'] == shape)]

        if 'solver' in df.columns and solver is not None:
            df = df[(df['solver'] == solver)]

        if 'initial condition' in df.columns:
            df = df[(df['initial condition'] == init_cond) | (df['shape'] != 'SPT')]

        if geometry == 'new':
            df = df[(df['maze dimensions'].isin(['MazeDimensions_human.xlsx',
                                                 'MazeDimensions_new2021_SPT_ant.xlsx',
                                                 'MazeDimensions_humanhand.xlsx']))]
            df = df[(df['load dimensions'].isin(['LoadDimensions_new2021_SPT_ant.xlsx',
                                                 'LoadDimensions_human.xlsx',
                                                 'LoadDimensions_humanhand.xlsx']))]
        else:
            if 'maze dimensions' in df.columns and geometry is not None:
                df = df[(df['maze dimensions'] == geometry[0])]

            if 'load dimensions' in df.columns and geometry is not None:
                df = df[(df['load dimensions'] == geometry[1])]

        if size is not None:
            if 'Small' in size:
                sizes = ['Small Far', 'Small Near']
            else:
                sizes = [size]
            df = df[(df['size'].isin(sizes))]

        if winner is not None:
            df = df[df['winner'] == winner]

        if communication is not None:
            df = df[(df['communication'] == communication)]

        if free is not None:
            # choose only the trajectories that have 'free' in the filename
            df = df[df['filename'].str.contains('free') == free]

        self.df = df

    def choose_columns(self, columns):
        self.df = self.df[columns]

    def get_seperate_filenames(self):
        filenames = {}
        for solver in ['ant', 'human']:
            dfss = self.get_separate_data_frames(solver=solver, plot_separately=plot_separately[solver],
                                                 shape='SPT', geometry=solver_geometry[solver], initial_cond='back')
            for size, dfs in dfss.items():
                for sep, df in dfs.items():
                    print(size, sep)
                    filenames[solver + ' ' + size + ' ' + sep] = list(df['filename'])
        return filenames

    def get_separate_data_frames(self, solver, plot_separately, shape=None, geometry=None, initial_cond='back',
                                 free=False) -> dict:
        # with open(averageCarrierNumber_dir, 'r') as json_file:
        #     averageCarrierNumber_dict = json.load(json_file)
        # self.df['average Carrier Number'] = self.df['filename'].map(averageCarrierNumber_dict)

        df = self.df[~self.df['free'].astype(bool)]
        if 'solver' in self.df.columns:
            df = df[df['solver'] == solver]
        if 'shape' in self.df.columns and shape is not None:
            df = df[df['shape'] == shape]
        if 'maze dimensions' in self.df.columns and geometry is not None:
            df = df[(df['maze dimensions'] == geometry[0])]
        if 'initial condition' in self.df.columns and initial_cond is not None:
            df = df[df['initial condition'] == initial_cond]

        df_to_plot = {}
        if solver == 'ant':
            def split_winners_loosers(size, df):
                winner = df[df['winner'] & (df['size'] == size)]
                looser = df[~df['winner'] & (df['size'] == size)]
                return {'winner': winner, 'looser': looser}

            df_add = {'XL': split_winners_loosers('XL', df),
                      'L': split_winners_loosers('L', df),
                      'M': split_winners_loosers('M', df),
                      'S (> 1)': split_winners_loosers('S',
                                                       df[~df['average Carrier Number'].isin(plot_separately['S'])]),
                      'Single (1)': split_winners_loosers('S',
                                                          df[df['average Carrier Number'].isin(plot_separately['S'])])}
            df_to_plot.update(df_add)

        if solver == 'pheidole':
            def split_winners_loosers(size, df):
                winner = df[df['winner'] & (df['size'] == size)]
                looser = df[~df['winner'] & (df['size'] == size)]
                return {'winner': winner, 'looser': looser}

            df_add = {'XL': split_winners_loosers('XL', df),
                      'L': split_winners_loosers('L', df),
                      'M': split_winners_loosers('M', df),
                      'S (> 1)': split_winners_loosers('S', df)}
            df_to_plot.update(df_add)

        if 'human' in solver.split('_'):
            def split_NC_C(sizes, df):
                communication = df[df['communication'] & (df['size'].isin(sizes))]
                non_communication = df[~df['communication'] & (df['size'].isin(sizes))]
                return {'communication': communication, 'non_communication': non_communication}

            df_add = {'Large': split_NC_C(['Large'], df),
                      'M (>7)': split_NC_C(['Medium'],
                                           df[~df['average Carrier Number'].isin(plot_separately['Medium'])]),
                      'M (2)': split_NC_C(['Medium'],
                                          df[df['average Carrier Number'].isin([plot_separately['Medium'][0]])]),
                      'M (1)': split_NC_C(['Medium'],
                                          df[df['average Carrier Number'].isin([plot_separately['Medium'][1]])]),
                      'Small': split_NC_C(['Small Far', 'Small Near'], df)
                      }
            df_to_plot.update(df_add)

        if 'humanhand' in solver.split('_'):
            e = ExcelSheet()
            with_eyesight, without_eyesight = [], []
            for filename in df['filename']:
                if filename[0] == 'Y':
                    if e.with_eyesight(filename):
                        with_eyesight.append(filename)
                    else:
                        without_eyesight.append(filename)

            def split_NC_C(sizes, df):
                with_eyesight_df = df[df['filename'].isin(with_eyesight)]
                without_eyesight_df = df[df['filename'].isin(without_eyesight)]
                return {'with_eyesight': with_eyesight_df, 'without_eyesight': without_eyesight_df, }

            df_add = {'': split_NC_C(None, df)}
            df_to_plot.update(df_add)
            # df_to_plot.update(split_NC_C(None, df))

        return df_to_plot

    def save_separate_excel(self):
        with open(averageCarrierNumber_dir, 'r') as json_file:
            averageCarrierNumber_dict = json.load(json_file)
        self.df['average Carrier Number'] = self.df['filename'].map(averageCarrierNumber_dict)

        dfss = self.get_separate_data_frames(solver='ant', plot_separately=plot_separately['ant'],
                                             initial_cond='front',
                                             geometry=('MazeDimensions_ant.xlsx',
                                                       'LoadDimensions_ant.xlsx'))
        for size, dfs in dfss.items():
            for sep, df in dfs.items():
                size_str = size.replace('>', 'more than')
                df = find_directory_of_original(df)
                df.to_excel(lists_exp_dir + '\\exp_ant_' + size_str + '_' + sep + '_old.xlsx')
        DEBUG = 1

        # dfss = self.get_separate_data_frames(solver='ant', plot_separately=plot_separately['ant'],
        #                                      initial_cond='back',
        #                                      geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
        #                                                'LoadDimensions_new2021_SPT_ant.xlsx'))
        # for size, dfs in dfss.items():
        #     for sep, df in dfs.items():
        #         size_str = size.replace('>', 'more than')
        #         df.to_excel(lists_exp_dir + '\\exp_ant_' + size_str + '_' + sep + '.xlsx')
        #
        # self.df.to_excel(lists_exp_dir + '\\exp.xlsx')
        # dfss = self.get_separate_data_frames(solver='human', plot_separately=plot_separately['human'])
        # for size, dfs in dfss.items():
        #     for communcation in ['communication', 'non_communication']:
        #         size_str = size.replace('>', 'more than ')
        #         dfs[communcation].to_excel(lists_exp_dir + '\\exp_human_' + size_str + '_' + communcation + '.xlsx')
        #
        # dfss = self.get_separate_data_frames(solver='humanhand', plot_separately=plot_separately['humanhand'])
        # for winner, dfs in dfss.items():
        #     for communcation in ['with_eyesight', 'without_eyesight']:
        #         dfs[communcation].to_excel(lists_exp_dir + '\\exp_humanhand_' + communcation + '.xlsx')
        #
        # dfss = self.get_separate_data_frames(solver='pheidole', plot_separately=plot_separately['pheidole'])
        # for size, dfs in dfss.items():
        #     for sep, df in dfs.items():
        #         size_str = size.replace('>', 'more than')
        #         dfs[sep].to_excel(lists_exp_dir + '\\exp_pheidole_' + size_str + '_' + sep + '_.xlsx')
        #
        # ad.choose_experiments(free=True)
        # self.df.to_excel(lists_exp_dir + '\\exp_free.xlsx')


def find_directory_of_original(df):
    def find_dir(name, solver):
        # name = 'large_20201220135801_20201220140247'
        if solver == 'human':
            for root, dirs, files in os.walk(original_movies_dir_human):
                for dir in [hu for hu in dirs if hu.startswith('2')]:
                    for size in os.listdir(os.path.join(root, dir, 'Videos')):
                        human_dir = os.path.join(root, dir, 'Videos', size)
                        movies = [n for n in os.listdir(human_dir) if (n.endswith('.asf') and n.startswith('NVR'))]
                        if name.split('_')[1] in [n.split('_')[3] for n in movies]:
                            index = np.where(name.split('_')[1] ==
                                             np.array([n.split('_')[3] for n in movies]))[0][0]
                            address = os.path.join(human_dir, movies[index])
                            # print(address)
                            return address

        elif solver in ['ant', 'pheidole']:
            for direct in original_movies_dir_ant:
                for root, dirs, files in os.walk(direct):
                    for dir in dirs:
                        if name.split('_')[2] in [n[1:8] for n in os.listdir(os.path.join(root, dir))]:
                            index = np.where(name.split('_')[2] ==
                                             np.array([n[1:8] for n in os.listdir(os.path.join(root, dir))]))[0][0]
                            # print(os.path.join(root, dir, os.listdir(os.path.join(root, dir))[index]))
                            if 'S4080004.MP4' in \
                                os.path.join(root, dir, os.listdir(os.path.join(root, dir))[index]):
                                DEBUG = 1
                            if 'Special' in os.path.join(root, dir, os.listdir(os.path.join(root, dir))[index]) or \
                                    'SPT' in os.path.join(root, dir, os.listdir(os.path.join(root, dir))[index]):
                                return os.path.join(root, dir, os.listdir(os.path.join(root, dir))[index])
        if solver == 'humanhand':
            return os.path.join(original_movies_dir_humanhand, name + '.MP4')
        print('not found ', name)
        return None
    df['directory'] = df[['filename', 'solver']].apply(lambda x: find_dir(x['filename'], x['solver']), axis=1)
    return df


def choose_trajectories(solver='human', size='Large', shape='SPT',
                        geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                        communication=None, number: int = None, init_cond: str = 'back') \
        -> list:
    df = Altered_DataFrame()
    df.choose_experiments(solver=solver, shape=shape, geometry=geometry, size=size, communication=communication,
                          init_cond=init_cond)
    filenames = df.df.filename[:number]

    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    # bad_filename = 'S_SPT_4750002_SSpecialT_1_ants (part 1)' # TODO: fix this trajectory
    return [get(filename) for filename in filenames]


if __name__ == '__main__':
    ad = Altered_DataFrame()
    # ad = find_directory_of_original(ad)
    ad.save_separate_excel()

    DEBUG = 1
    # ad.choose_experiments(solver='ant', size='S', shape='SPT', init_cond='back', winner=True)
