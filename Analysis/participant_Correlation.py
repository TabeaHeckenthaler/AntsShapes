from matplotlib import pyplot as plt
import numpy as np
from trajectory_inheritance.trajectory import get
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from trajectory_inheritance.humans import angle_shift

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

# f0 = x.participants.forces.force_vector(0, reference_frame='maze')
# f1 = x.participants.forces.force_vector(1, reference_frame='maze')

# fig, ax = plt.subplots()
#
# frame = 1120
#
# for name in [0, 4]:
#     f = x.participants.forces.force_vector(name, reference_frame='load')
#     rot = np.cross(x.participants.forces.meters_load[name], f, axisa=0, axisb=0)
#     ax.plot(rot)
#     ax.plot(frame, rot[frame], '*')
# plt.show()
# plot_correlation(x)


class Participant_Correlation:
    def __init__(self, x):
        self.x = x
        if not hasattr(self.x, 'participants'):
            x.load_participants()

    def correlation(self, players=None, frames=None) -> np.ndarray:
        """
        :param frames: forces in what frames are you interested in finding their correlation
        :param players: list of players that you want to find correlation for
        :return: nxn correlation matrix, where n is either the length of the kwarg players or the number of forcemeters
        on the shape
        """
        if frames is None:
            frames = [0, len(self.x.frames)]
        if players is None:
            players = self.x.occupied

        forces_in_x_direction = [
            self.x.forces.abs_values[:, player][slice(*frames, 1)] * np.cos(angle_shift[self.x.size][player])
            for player in players]
        correlation_matrix = np.corrcoef(np.stack(forces_in_x_direction))

        fig, ax = plt.subplots()
        im = ax.imshow(np.triu(correlation_matrix), cmap='seismic_r')

        ax.axes.xaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
        ax.axes.yaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
        im.set_clim(-1, 1)
        fig.colorbar(im)
        fig.show()
        return correlation_matrix

    def plot_abs_values(self):
        plt.plot(self.x.frames / self.x.fps, x.participants.forces.abs_values)
        plt.show()

    def plot_torque_correlation(self) -> None:
        torques = np.stack([self.x.participants.forces.torque(part) for part in x.participants.occupied], axis=1)
        Display(self.x, Maze(self.x), i=1500).draw(x)
        plt.plot(torques)
        plt.plot(np.mean(torques, axis=1))
        plt.legend(self.x.participants.occupied)
        plt.show()


if __name__ == '__main__':
    x = get('medium_20201223125622_20201223130532')
    cor = Participant_Correlation(x)
    cor.plot_torque_correlation()
