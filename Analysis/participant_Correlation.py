from matplotlib import pyplot as plt
import numpy as np
from trajectory_inheritance.trajectory import get
from Setup.Maze import Maze
from PhysicsEngine.Display import Display

# something old
# def plot_correlation(x, frames=None) -> None:
#     x.load_participants()
#     corr = x.participants.correlation(frames=frames)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(np.triu(corr), cmap='seismic_r')
#     ax.axes.xaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
#     ax.axes.yaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
#     im.set_clim(-1, 1)
#     fig.colorbar(im)
#     fig.show()

# f0 = x.participants.forces.part(0, reference_frame='maze')
# f1 = x.participants.forces.part(1, reference_frame='maze')

# fig, ax = plt.subplots()
#
# frame = 1120
#
# for name in [0, 4]:
#     f = x.participants.forces.part(name, reference_frame='load')
#     rot = np.cross(x.participants.forces.meters_load[name], f, axisa=0, axisb=0)
#     ax.plot(rot)
#     ax.plot(frame, rot[frame], '*')
# plt.show()
# plot_correlation(x)


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
    x.load_participants()
    plt.plot(x.participants.forces.abs_values)
    cor = Participant_Correlation(x)
    cor.torque([0, 5], 0)

    Display(x, cor.maze).draw(x)
    k = 1
