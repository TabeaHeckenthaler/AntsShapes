import pandas as pd
from trajectory_inheritance.get import get

length_unit = {'ant': 'cm', 'human': 'm', 'humanhand': 'cm', 'ps_simulation': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


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
        self['communication'] = bool(x.communication)
        self['length unit'] = str(length_unit_func(x.solver))
        self['initial condition'] = str(x.initial_cond())
        self['force meter'] = bool(x.has_forcemeter())
        self['maze dimensions'], self['load dimensions'] = x.geometry()
        self['counted carrier number'] = None
        self['time [s]'] = x.timer()

        # from Analysis.PathLength.PathLength import PathLength
        # from Setup.Attempts import Attempts
        # self['average Carrier Number'] = averageCarrierNumber[x.filename]
        # self['average Carrier Number'] = float(x.averageCarrierNumber())
        # self['Attempts'] = Attempts(x, 'extend')
        # self['path length [length unit]'] = float(PathLength(x).per_experiment(penalize=False))
        # self['minimal path length [length unit]'] = PathLength(x).minimal()
        # self['penalized path length [length unit]'] = float(PathLength(x).per_experiment(penalize=True))
        # if x.shape != 'SPT':
        #     self['path length during attempts [length unit]'] = PathLength(x).during_attempts()
        # self['solving time [s]'] = x.solving_time()
        # self = self[list_of_columns]