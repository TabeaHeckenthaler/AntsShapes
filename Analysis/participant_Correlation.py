from matplotlib import pyplot as plt
import numpy as np
from trajectory_inheritance.trajectory import get
from Setup.Maze import Maze
from PhysicsEngine.Display import Display


class Participant_Correlation:
    def __init__(self, x):
        self.x = x
        self.maze = Maze(x, )
        if not hasattr(self.x, 'participants'):
            x.load_participants()
        self.forcemeters = self.maze.force_attachment_positions_in_trajectory(x)[:, self.x.participants.occupied, :]
        self.force_vectors = self.x.participants.forces.part(self.x.participants.occupied).reshape(self.forcemeters.shape)

    def torque(self, participant_list, i):
        for part in participant_list:
            t = np.cross(self.force_vectors[i, part], self.forcemeters[i, part])
            print(t)


if __name__ == '__main__':
    x = get('medium_20201223125622_20201223130532')
    cor = Participant_Correlation(x)
    cor.torque([0, 5], 0)

    Display(x, cor.maze).draw(x)
    k = 1
