from trajectory_inheritance.trajectory import Trajectory
import numpy as np
from trajectory_inheritance.gillespie import Gillespie, N_max
from Setup.Load import Load
from PhysicsEngine.mainGame import mainGame
from Setup.Maze import Maze

time_step = 0.01


# def newFileName(size, shape):
#     counter = int(len(glob.glob(size + '_' + shape + '*_' + '_gillespie_' + '_*')) / 2 + 1)
#     filename = size + '_' + shape + '_gillespie_' + str(counter)
#     return filename


class Trajectory_gillespie(Trajectory):
    def __init__(self, size=None, shape=None, filename='gillespie_test', fps=time_step, winner=bool, free=False):

        solver = 'gillespie'
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.free = free

        my_maze = Maze(size=size, shape=shape, solver=solver, free=free)
        my_load = Load(my_maze)

        self.gillespie = Gillespie(my_load)

    def step(self, my_load, i, my_maze, display):

        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]

        if self.gillespie.time_until_next_event < time_step:
            self.gillespie.time_until_next_event = self.gillespie.whatsNext(my_load)

        self.forces(my_load, display=display)

        self.gillespie.time_until_next_event -= time_step
        my_maze.Step(time_step, 10, 10)

        self.position = np.vstack((self.position, [my_load.position.x, my_load.position.y]))
        self.angle = np.hstack((self.angle, my_load.angle))
        return

    def forces(self, my_load, pause=False, display=None):
        my_load.linearVelocity = 0 * my_load.linearVelocity
        my_load.angularVelocity = 0 * my_load.angularVelocity

        """ Magnitude of forces """
        display.arrows = []

        for i in range(len(self.gillespie.n_p)):
            start = self.gillespie.attachment_site_world_coord(my_load, i)
            end = None

            if self.gillespie.n_p[i]:
                f_x, f_y = self.gillespie.ant_force(my_load, i, pause=pause)

                start = start
                end = start + [10 * f_x, 10 * f_y]
                display.arrows.append((start, end, 'puller'))

            elif self.gillespie.n_l[i]:
                display.arrows.append((start, end, 'lifter'))

            else:
                display.arrows.append((start, end, 'empty'))

    def run_simulation(self, frameNumber, display=True, free=False, wait=0):
        my_maze = Maze(size=self.size, shape=self.shape, solver='sim', free=free)
        self.frames = np.linspace(1, frameNumber, frameNumber)
        self.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
        self.angle = np.array([0], dtype=float)  # array to store the position and angle of the load
        mainGame(self, display=display, free=free, wait=wait)

    def participants(self):
        return self.gillespie

    def averageCarrierNumber(self):
        return N_max  # TODO: this is maximum, not average...


if __name__ == '__main__':
    # possible shapes: 'I', 'circle', 'SPT' (only here, Gillespie is implemented)
    # free is either False or True
    # size only relevant for 'I': (XS, S, M, SL, L, XL) and 'SPT' (S, M, L, XL)

    x = Trajectory_gillespie(size='L', shape='SPT', filename='gillespie_trajectory1')
    x.run_simulation(frameNumber=500)

    x.play()  # you can play the trajectory again!
    x.save()
