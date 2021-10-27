from trajectory_inheritance.trajectory import get
from matplotlib import pyplot as plt
import numpy as np


def plot_correlation(x, frames=None) -> None:
    x.load_participants()
    corr = x.participants.correlation(frames=frames)

    fig, ax = plt.subplots()
    im = ax.imshow(np.triu(corr), cmap='seismic_r')
    ax.axes.xaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
    ax.axes.yaxis.set(ticks=np.arange(0, len(x.participants.occupied)), ticklabels=x.participants.occupied)
    im.set_clim(-1, 1)
    fig.colorbar(im)
    fig.show()


# x = get('medium_20210421115444_20210421115951', 'human')
# x.load_participants()
# plot_correlation(x, frames=[2100, 2300])

if __name__ == '__main__':
    x = get('large_20201220135801_20201220140247')
    x.load_participants()
    x.play()

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
