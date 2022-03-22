from Analysis.PathPy.Network import *
from DataFrame.choose_experiments import choose_experiments


def calculate_diffusion_time(filenames, paths, solver, size, shape) -> pd.Series:
    paths.load_paths(filenames=filenames)

    my_network = Network.init_from_paths(paths, solver, size, shape)
    my_network.get_results()

    t = my_network.t.sort_values(0, ascending=False) * paths.time_step  # TODO: this easily leads
    # to mistakes
    return t


class DiffusionTime:
    def __init__(self, solver, shape, geometry):
        self.solver, self.shape, self.geometry = solver, shape, geometry

    def get_diffusion_times(self) -> dict:
        pass

    def plot(self):
        fig, ax = plt.subplots()
        legend = []
        diff_times = self.get_diffusion_times()

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
        fig.savefig(os.path.join(graph_dir(), self.saving_name() + '.png'),
                    format='png', pad_inches=0.5, bbox_inches='tight')

    def y_label(self):
        pass

    def saving_name(self) -> str:
        pass


class DiffusionTimeHuman(DiffusionTime):
    def __init__(self):
        super().__init__('human', 'SPT', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

    def y_label(self):
        return 'time expected to pass before humans solve [s]'

    def saving_name(self):
        return 'human_expected_solving_time'

    def get_diffusion_times(self):
        diff_times = {}
        for i, self.size in enumerate(exp_types[self.shape][self.solver]):
            for communication in [True, False]:
                paths = Paths(self.solver, self.size, self.shape, self.geometry)
                filenames = choose_experiments(self.solver, self.shape, self.size, self.geometry,
                                               communication=communication).filename
                if len(filenames) > 0:
                    t = calculate_diffusion_time(filenames, paths, self.solver, self.size, self.shape)
                    diff_times[self.size + '_comm_' + str(communication)] = t
        return diff_times


class DiffusionTimeAnt(DiffusionTime):
    def __init__(self):
        super().__init__('ant', 'SPT', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    def y_label(self):
        return 'time expected to pass before ants solve [s]'

    def saving_name(self):
        return 'ants_expected_solving_time'

    def get_diffusion_times(self):
        diff_times = {}
        for i, size in enumerate(exp_types[self.shape][self.solver]):
                paths = Paths(self.solver, size, self.shape, self.geometry)
                filenames = choose_experiments(self.solver, self.shape, size, self.geometry).filename
                if len(filenames) > 0:
                    t = calculate_diffusion_time(filenames, paths, self.solver, size, self.shape)
                    diff_times[size] = t
        return diff_times


if __name__ == '__main__':
    diff_time_human = DiffusionTimeHuman()
    diff_time_human.plot()

    diff_time_human = DiffusionTimeAnt()
    diff_time_human.plot()
