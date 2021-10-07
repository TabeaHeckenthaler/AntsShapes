from trajectory_inheritance.trajectory import Trajectory


class Trajectory_ps_simulation(Trajectory):

    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool):
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.sensing = int()
        self.dilation = int()

    def participants(self):
        from Classes_Experiment.mr_dstar import Mr_dstar
        return Mr_dstar(self)

    def step(self, my_load, i, **kwargs):
        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]
