from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_humanhand import ExcelSheet
from trajectory_inheritance.exp_types import solver_geometry

plot_separately = {'ant': {'S': [1]}, 'human': {'Medium': [2, 1]}, 'humanhand': {'': []}}

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
            df = df[~(df['filename'].str.contains('free'))]

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

    def get_separate_data_frames(self, solver, plot_separately, shape=None, geometry=None, initial_cond='back'):
        if 'solver' in self.df.columns:
            df = self.df[self.df['solver'] == solver]
        else:
            df = self.df
        if 'shape' in self.df.columns and shape is not None:
            df = df[df['shape'] == shape]
        if 'maze dimensions' in self.df.columns and geometry is not None:
            df = df[(df['maze dimensions'] == geometry[0])]
        if 'initial condition' in self.df.columns:
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
    ad.choose_experiments(solver='ant', size='S', shape='SPT', init_cond='back', winner=True)
