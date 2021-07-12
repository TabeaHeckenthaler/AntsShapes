import numpy as np
from os import path, getcwd, mkdir
import datetime
from matplotlib import pyplot as plt
from trajectory import home


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def angles_shift(angle):
    # shift [0, 2pi] to [-pi, pi]
    angle1 = angle + np.pi
    angle1_mod = angle1 % (np.pi * 2) - np.pi
    return angle1_mod


colors = {'H': 'k', 'I': 'b', 'T': 'c', 'SPT': 'r',
          'RASH': 'm', 'LASH': 'm', 'ASH': 'm',
          'Small': 'k', 'Medium': 'b', 'Large': 'r',
          'XS': 'c', 'S': 'm', 'M': 'g', 'L': 'y', 'SL': 'tab:orange', 'XL': 'tab:pink',
          'human': 'blue', 'ant': 'red', 'dstar': 'black', 'humanhand': 'green',
          False: 'r', True: 'k'}


def graph_dir():
    direct = path.abspath(path.join(home, 'Graphs',
                                    datetime.datetime.now().strftime("%Y") + '_' +
                                    datetime.datetime.now().strftime("%m") + '_' +
                                    datetime.datetime.now().strftime("%d")))
    if not (path.isdir(direct)):
        mkdir(direct)
    return direct


def non_duplicate_legend(ax):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def three_D_plotting(x, y, z, zerr=None, yerr=None, xerr=None, label='', color='black', **kwargs):
    if 'ax' not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs['ax']
    errors = [xerr, yerr, zerr]
    for ii, error in enumerate([xerr, yerr, zerr]):
        if error is None:
            errors[ii] = np.zeros(x.shape)

    for xi, yi, zi, xerri, yerri, zerri in zip(x, y, z, *errors):
        ax.plot(xi, yi, zi,
                linestyle='',
                marker='o',
                c=color,
                label=label,
                )
        kwargs = {'marker': "_", 'c': color, 'label': ''}
        ax.plot([xi + xerri, xi - xerri], [yi, yi], [zi, zi], **kwargs)
        ax.plot([xi, xi], [yi + yerri, yi - yerri], [zi, zi], **kwargs)
        ax.plot([xi, xi], [yi, yi], [zi + zerri, zi - zerri], **kwargs)
    return


# x.tracked_frames = ranges(x.frames)
def ranges(nums, *args, **kwargs):
    if 'scale' in kwargs:
        scale = kwargs['scale']
    else:
        scale = np.linspace(0, len(nums) - 1, len(nums) - 0)

    if 'boolean' in args:
        nums = np.where(np.array(nums))[0]

    if 'smallest_gap' in kwargs:
        smallest_gap = kwargs['smallest_gap']
    else:
        smallest_gap = 2

    if 'buffer' in kwargs:
        buffer = kwargs['buffer']
    else:
        buffer = 0

    if len(nums) == 0:
        return []

    if len(nums) == 1:
        return [[scale[nums[0]], scale[nums[0] + 1]]]

    ran = [[scale[nums[0]], scale[nums[-1]]]]
    for i in range(len(nums) - 1):
        if smallest_gap < nums[i + 1] - nums[i]:
            ran[-1] = [ran[-1][0], scale[nums[i]] + 1 + buffer]
            ran.append([scale[nums[i + 1]] - buffer, scale[nums[-1]] + buffer])

    return ran


def read_text_file(directory, filename, **kwargs):
    with open(directory + path.sep + filename) as text_file:
        lines = text_file.readlines()
    return lines


def plot_boolean_shading(values, boolean_array):
    fig, ax = plt.subplots()
    plt.plot(values)
    ax.fill_between(np.linspace(0, ax.get_xlim()[1], len(boolean_array)), ax.get_ylim()[0], ax.get_ylim()[1],
                    where=boolean_array, alpha=0.4)
