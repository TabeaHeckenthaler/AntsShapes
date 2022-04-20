from Analysis.PathPy.Network import *
from DataFrame.Altered_DataFrame import Altered_DataFrame


class Diffusion:
    def __init__(self, solver, shape, geometry):
        self.solver, self.shape, self.geometry = solver, shape, geometry

    def get_diffusion(self) -> dict:
        pass

    def y_label(self):
        pass

    def saving_name(self) -> str:
        pass

    def plot(self):
        fig, ax = plt.subplots()
        legend = []
        diff_times = self.get_diffusion()

        index = sorted(list({x for l in [list(t.index) for t in diff_times.values()] for x in l}))
        ax.set_xticks(ticks=range(len(index)))
        ax.set_xticklabels(index)

        for key, t in diff_times.items():
            t = t.reindex(index)
            t.plot(ax=ax)
            legend.append(key)
        plt.show(block=False)
        ax.set_ylabel(self.y_label())
        ax.legend(legend)
        print('Saving figure in ' + os.path.join(graph_dir(), self.saving_name() + '.png'))
        fig.savefig(os.path.join(graph_dir(), self.saving_name() + '.png'),
                    format='png', pad_inches=0.5, bbox_inches='tight')


class DiffusionTime(Diffusion):
    def __init__(self, solver, shape, geometry):
        super().__init__(solver, shape, geometry)

    def calculate_diffusion_time(self, filenames, size) -> pd.Series:
        paths = Paths(self.solver, size, self.shape, self.geometry)
        paths.load_paths(filenames=filenames)

        my_network = Network.init_from_paths(paths, self.solver, size, self.shape)
        my_network.get_results()
        t = my_network.t.sort_values(0, ascending=False) * paths.time_step  # TODO: this easily leads to mistakes
        return t


class DiffusionTimeHuman(DiffusionTime):
    def __init__(self):
        super().__init__('human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

    def y_label(self):
        return 'time expected to pass before humans solve [s]'

    def saving_name(self):
        return 'human_expected_solving_time'

    def get_diffusion(self):
        diff_times = {}
        for i, size in enumerate(exp_types[self.shape][self.solver]):
            for communication in [True, False]:
                df = Altered_DataFrame()
                filenames = df.choose_experiments(self.solver, self.shape, self.geometry, size=size,
                                                  communication=communication).df.filename
                if len(filenames) > 0:
                    diff_times[size + '_comm_' + str(communication)] = \
                        self.calculate_diffusion_time(filenames, size)
        return diff_times


class DiffusionTimeAnt(DiffusionTime):
    def __init__(self):
        super().__init__('ant', 'SPT', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    def y_label(self):
        return 'time expected to pass before ants solve [s]'

    def saving_name(self):
        return 'ants_expected_solving_time'

    def get_diffusion(self):
        diff_times = {}
        for i, size in enumerate(exp_types[self.shape][self.solver]):
            df = Altered_DataFrame()
            filenames = df.choose_experiments(self.solver, self.shape, self.geometry, size=size).df.filename
            if len(filenames) > 0:
                diff_times[size] = self.calculate_diffusion_time(filenames, size)
        return diff_times


class DiffusionStates(Diffusion):
    def __init__(self, solver, shape, geometry):
        super().__init__(solver, shape, geometry)

    def calculate_diffusion_states(self, filenames, size) -> pd.Series:
        paths = PathWithoutSelfLoops(self.solver, size, self.shape, self.geometry)
        paths.load_paths(filenames=filenames)
        my_network = Network.init_from_paths(paths, self.solver, size, self.shape)
        my_network.get_results()
        # my_network.save(my_network.to_dict())
        t = my_network.t.sort_values(0, ascending=False)
        return t


class DiffusionStatesHuman(DiffusionStates):
    def __init__(self):
        super().__init__('human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

    def y_label(self):
        return 'states passed before humans solve [s]'

    def saving_name(self):
        return 'human_expected_solving_states'

    def get_diffusion(self):
        diff_times = {}
        for i, size in enumerate(exp_types[self.shape][self.solver]):
            for communication in [True, False]:
                df = Altered_DataFrame()
                filenames = df.choose_experiments(self.solver, self.shape, self.geometry, size=size,
                                                  communication=communication).df.filename
                if len(filenames) > 0:
                    diff_times[size + '_comm_' + str(communication)] = \
                        self.calculate_diffusion_states(filenames, size)
        return diff_times


class DiffusionStatesAnt(DiffusionStates):
    def __init__(self):
        super().__init__('ant', 'SPT', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    def y_label(self):
        return 'states passed before ants solve [s]'

    def saving_name(self):
        return 'ants_expected_solving_states'

    def get_diffusion(self):
        diff_times = {}
        for i, size in enumerate(exp_types[self.shape][self.solver]):
            df = Altered_DataFrame()
            filenames = df.choose_experiments(self.solver, self.shape, self.geometry, size=size).df.filename
            if len(filenames) > 0:
                diff_times[size] = self.calculate_diffusion_states(filenames, size)
        return diff_times


if __name__ == '__main__':
    diff_time_human = DiffusionTimeHuman()
    diff_time_human.plot()

    diff_time_ant = DiffusionTimeAnt()
    diff_time_ant.plot()

    diff_state_human = DiffusionStatesHuman()
    diff_state_human.plot()

    diff_state_ant = DiffusionStatesAnt()
    diff_state_ant.plot()
    DEBUG = 1