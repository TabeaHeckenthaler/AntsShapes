from trajectory_inheritance.trajectory import Trajectory


def filename_dstar(size, shape, dil_radius, sensing_radius):
    return size + '_' + shape + '_' + 'dil' + str(dil_radius) + '_sensing' + str(sensing_radius)


class Trajectory_ps_simulation(Trajectory):

    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool):
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.sensing = int()
        self.dilation = int()

    def participants(self):
        return PS_simulation(self)

    def step(self, my_load, i, **kwargs):
        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]

    def averageCarrierNumber(self):
        return 1


class PS_simulation:
    def __init__(self, filename):
        self.filename = filename
        return