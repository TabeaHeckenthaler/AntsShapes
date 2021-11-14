from matplotlib import pyplot as plt
import numpy as np
from trajectory_inheritance.trajectory import get
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from trajectory_inheritance.humans import angle_shift, participant_number
import pandas as pd
from DataFrame.plot_dataframe import save_fig
from tqdm import tqdm


class Participant_Correlation:
    def __init__(self, x):
        self.x = x
        if not hasattr(self.x, 'participants'):
            x.load_participants()

    def linear_correlation(self, players=None, frames=None) -> pd.DataFrame:
        """
        :param frames: forces in what frames are you interested in finding their correlation
        :param players: list of players that you want to find correlation for
        :return: nxn correlation matrix, where n is either the length of the kwarg players or the number of forcemeters
        on the shape
        """
        if frames is None:
            frames = [0, len(self.x.frames)]
        if players is None:
            players = range(participant_number[self.x.size])

        forces_in_x_direction = [self.x.participants.forces.abs_values[:, player][slice(*frames, 1)] *
                                 np.cos(angle_shift[self.x.size][player])
                                 for player in players]

        forces_in_x_direction = pd.DataFrame(np.stack(forces_in_x_direction, axis=1))
        correlation_matrix = forces_in_x_direction.corr()
        return correlation_matrix

    def plot_abs_values(self):
        plt.plot(self.x.frames / self.x.fps, self.x.participants.forces.abs_values)
        plt.show()

    def torque_correlation(self) -> pd.DataFrame:
        """
        Calculate correlation between all the torques exerted by the participants
        :return : matrix with correlation values of all the occupied sites.
        """
        torques = pd.DataFrame(np.transpose([self.x.participants.forces.torque(part)
                                             for part in range(participant_number[self.x.size])]))

        # ax = torques.plot()
        # ax1 = torques.mean(axis=1).plot()
        return torques.corr()


def plot_correlation(torques_corr: np.array, ticklabels) -> plt.figure:
    """
    plot correlation in seismic triangle, so only upper half is filled.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(np.triu(torques_corr), cmap='seismic_r')
    ax.axes.xaxis.set(ticks=ticklabels, ticklabels=ticklabels)
    ax.axes.yaxis.set(ticks=ticklabels, ticklabels=ticklabels)
    im.set_clim(-1, 1)
    fig.colorbar(im)
    return fig


def forcemeter_itterator():
    from DataFrame.dataFrame import myDataFrame
    sizes = ['Large', 'Medium']
    for size in sizes:
        print(size)
        for com in [True, False]:
            print('communication: ' + str(com))
            df = myDataFrame[(myDataFrame['communication'] == com) &
                             (myDataFrame['size'] == size) &
                             (myDataFrame['force meter'])]
            return df, size, com


def average_torque_correlation() -> None:
    """
    plot the average correlation for Medium and large sizes of
    """
    for df, size, com in forcemeter_itterator():
        correlations = []
        for index in tqdm(df.index):
            x = get(df.loc[index]['filename'])
            correlations.append(Participant_Correlation(x).torque_correlation())

        fig = plot_correlation(np.nanmean(np.array(correlations), axis=0),
                               ticklabels=range(participant_number[size]))
        save_fig(fig, size + '_torque_correlation_' + str(com))


def average_linear_correlation() -> None:
    """
    plot the average correlation for Medium and large sizes of
    """
    for df, size, com in forcemeter_itterator():
        correlations = []
        for index in tqdm(df.index):
            x = get(df.loc[index]['filename'])
            correlations.append(Participant_Correlation(x).linear_correlation())

        fig = plot_correlation(np.nanmean(np.array(correlations), axis=0),
                               ticklabels=range(participant_number[size]))
        save_fig(fig, size + '_linear_correlation_' + str(com))


if __name__ == '__main__':
    # average_torque_correlation()
    average_linear_correlation()
    # x = get('medium_20201223125622_20201223130532')
    # x = get('large_20210726231259_20210726232557')
    # this top one one seems to have really beautiful correlation for neighboring pullers
    # x = get('large_20210726224634_20210726225802')

    # display = Display(self.x, Maze(self.x), i=1000)
    # display.draw(self.x)
    # display.end_screen()
