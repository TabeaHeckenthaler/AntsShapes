from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get


class Altered_DataFrame:
    def __init__(self):
        self.df = myDataFrame.clone()

    def choose_experiments(self, solver, shape, geometry, size=None, communication=None, init_cond: str = 'back', winner: bool = None):
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
        if 'shape' in df.columns:
            df = df[(df['shape'] == shape)]

        if 'solver' in df.columns:
            df = df[(df['solver'] == solver)]

        if 'initial condition' in df.columns:
            df = df[(df['initial condition'] == init_cond)]

        if 'maze dimensions' in df.columns:
            df = df[(df['maze dimensions'] == geometry[0])]

        if 'load dimensions' in df.columns:
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
        self.df = df

    def choose_columns(self, columns):
        self.df = self.df[columns]

    def get_separate_data_frames(self, solver, plot_seperately, shape=None, geometry=None, initial_cond='back'):
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
        if solver == 'ant':
            def split_winners_loosers(size, df):
                winner = df[df['winner'] & (df['size'] == size)]
                looser = df[~df['winner'] & (df['size'] == size)]
                return {'winner': winner, 'looser': looser}

            df_to_plot = {'XL': split_winners_loosers('XL', df),
                          'L': split_winners_loosers('L', df),
                          'M': split_winners_loosers('M', df),
                          'S (> 1)': split_winners_loosers('S', df[~df['average Carrier Number'].isin(plot_seperately['S'])]),
                          'Single (1)': split_winners_loosers('S', df[df['average Carrier Number'].isin(plot_seperately['S'])])}

        elif solver == 'human':
            def split_NC_C(sizes, df):
                communication = df[df['communication'] & (df['size'].isin(sizes))]
                non_communication = df[~df['communication'] & (df['size'].isin(sizes))]
                return {'communication': communication, 'non_communication': non_communication}

            df_to_plot = {'Large': split_NC_C(['Large'], df),
                          'M (>7)': split_NC_C(['Medium'], df[~df['average Carrier Number'].isin(plot_seperately['Medium'])]),
                          'M (2)': split_NC_C(['Medium'], df[df['average Carrier Number'].isin([plot_seperately['Medium'][0]])]),
                          'M (1)': split_NC_C(['Medium'], df[df['average Carrier Number'].isin([plot_seperately['Medium'][1]])]),
                          'Small': split_NC_C(['Small Far', 'Small Near'], df)
                          }
        else:
            raise ValueError('What type of solver?')
        return df_to_plot


def choose_trajectories(solver='human', size='Large', shape='SPT',
                        geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                        communication=None, number: int = None, init_cond: str ='back') \
        -> list:
    df = Altered_DataFrame()
    df.choose_experiments(solver, shape, geometry, size=size, communication=communication,
                          init_cond=init_cond)
    filenames = df.df.filename[:number]

    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    # bad_filename = 'S_SPT_4750002_SSpecialT_1_ants (part 1)' # TODO: fix this trajectory
    return [get(filename) for filename in filenames]
