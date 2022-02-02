from trajectory_inheritance.trajectory import Trajectory


def filename_dstar(size, shape, dil_radius, sensing_radius):
    return size + '_' + shape + '_' + 'dil' + str(dil_radius) + '_sensing' + str(sensing_radius)


class Trajectory_ps_simulation(Trajectory):

    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, geometry=None):
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.sensing = int()
        self.dilation = int()

        # to not overwrite names
        if geometry is None:
            self.g = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
        else:
            self.g = geometry

    def load_participants(self):
        self.participants = PS_simulation(self)

    def step(self, my_maze, i, **kwargs):
        my_maze.set_configuration(self.position[i], self.angle[i])
        # load.position.x, load.position.y, load.angle = self.position[i][0], self.position[i][1], self.angle[i]

    def averageCarrierNumber(self):
        return 1

    def geometry(self):
        # At the moment I am only using the human dimensions
        return self.g


class PS_simulation:
    def __init__(self, filename):
        self.filename = filename