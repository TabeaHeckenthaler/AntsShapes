from trajectory_inheritance.get import get
from os import path
import json
import numpy as np
from Directories import network_dir
from DataFrame.import_excel_dfs import df_ant as df_ant_all
from copy import copy
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "Times New Roman"
maze_slits1 = {'XL': 13.02, 'L': 6.51, 'M': 3.255, 'S (> 1)': 1.6275, 'XS': 0.81375}
maze_slits2 = {'XL': 18.85, 'L': 9.53, 'M': 4.79, 'S (> 1)': 3.42, 'XS': 1.178}  # exp values, except for XS
maze_height = {'XL': 19.1, 'L': 9.55, 'M': 4.775, 'S (> 1)': 2.3875, 'XS': 1.19375}

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()


def extend_time_series_to_match_frames(ts, frame_len):
    indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                         for _ in range(frame_len)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def find_exp_day(filename):
    return filename.split('_')[2][:3]


def find_sign_change(x):
    return np.where(np.diff(np.sign(x)))[0]


def traj_followed_bias_flipping_ab(x):
    x = copy(x)
    x.smooth(2)

    inds = min(np.where(x.position[:, 0] < maze_slits1[size]))
    if len(inds) > 0:
        argmin = inds[0]
    else:
        print('never exited slit')
        return None

    y = x.position[argmin:, 1] - maze_height[size] / 2

    # most biased direction
    argmins2 = np.where(abs(y) > maze_height[size] / 5)[0] + argmin

    if argmins2.size > 0:
        argmin2 = argmins2[0]
        movement_direction_in_y = x.position[argmin2, 1] - x.position[0, 1]

        if movement_direction_in_y < 0:
            return 'right'
        elif movement_direction_in_y > 0:
            return 'left'

        # plt.plot(x.position[argmin:, 0], y)
        # plt.axvline(x=maze_slits[size], color='r')
        # plt.axhline(y=maze_height[size] * 1/5, color='r')
        # plt.axhline(y=-maze_height[size] * 1/5, color='r')
        #
        # # draw an arrow from x.position[argmin] to x.position[argmin2] - maze_height[size] / 2
        # plt.arrow(x.position[argmin, 0], x.position[argmin, 1] - maze_height[size] / 2,
        #           x.position[argmin2, 0] - x.position[argmin, 0],
        #           x.position[argmin2, 1] - x.position[argmin, 1],
        #           head_width=0.3, head_length=0.2, fc='k', ec='k')

    else:
        return None


def traj_followed_bias_flipping_c(x):
    x = copy(x)
    x.smooth(2)

    # entering slit
    inds = min(np.where(x.position[:, 0] > np.mean([maze_slits1[size], maze_slits2[size]])))
    if len(inds) > 0:
        argmin = inds[0]
    else:
        print('never exited slit')
        return None

    y = x.position[argmin:, 1] - maze_height[size] / 2
    if np.mean(y[:x.fps * 5]) < 0:
        return 'right'  # looking from the back of the maze, turning to the right is in the negative y direction
    return 'left'  # looking from the back of the maze, turning to the left is in the negative y direction

    # plt.plot(x.position[argmin:, 0], y)
    # plt.axvline(x=maze_slits1[size], color='r')
    # plt.axhline(y=maze_height[size] * 1/5, color='r')
    # plt.axhline(y=-maze_height[size] * 1/5, color='r')
    #
    # # draw an arrow from x.position[argmin] to x.position[argmin2] - maze_height[size] / 2
    # plt.arrow(x.position[argmin, 0], x.position[argmin, 1] - maze_height[size] / 2,
    #           x.position[argmin2, 0] - x.position[argmin, 0],
    #           x.position[argmin2, 1] - x.position[argmin, 1],
    #           head_width=0.3, head_length=0.2, fc='k', ec='k')


def traj_followed_bias_flipping_f(x):
    x = copy(x)
    x.smooth(2)

    # entering slit
    inds = min(np.where(x.position[:, 0] > maze_slits2[size] + (maze_slits2[size] - maze_slits1[size])/3))
    if len(inds) > 0:
        argmin = inds[0]
    else:
        print('never exited slit')
        return None

    y = x.position[argmin:, 1] - maze_height[size] / 2
    if np.mean(y) < 0:
        return 'right'  # looking from the back of the maze, turning to the right is in the negative y direction
    return 'left'  # looking from the back of the maze, turning to the left is in the negative y direction

    plt.plot(x.position[:, 0], x.position[:, 1])
    plt.plot(x.position[argmin:, 0], y)
    plt.plot(x.position[argmin:argmin + 5 * x.fps, 0], y[:5 * x.fps])
    plt.axvline(x=maze_slits1[size], color='r')
    plt.axvline(x=maze_slits2[size], color='r')
    #
    # # draw an arrow from x.position[argmin] to x.position[argmin2] - maze_height[size] / 2
    # plt.arrow(x.position[argmin, 0], x.position[argmin, 1] - maze_height[size] / 2,
    #           x.position[argmin2, 0] - x.position[argmin, 0],
    #           x.position[argmin2, 1] - x.position[argmin, 1],
    #           head_width=0.3, head_length=0.2, fc='k', ec='k')


def reduce_legend(ax):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def plot_biases(followed_bias_ab, followed_bias_c, followed_bias_f, ax, size):
    # go through every exp_day and plot the percentage of 'right' and 'left' followed biases
    colors_bias = {'right': 'y', 'left': 'cyan', None: 'r'}
    for exp_day, df_ant_day in exp_days(size):
        bottom = 0
        for i, filename in enumerate(df_ant_day['filename']):
            if filename not in followed_bias_ab:
                ax.bar(exp_day + '_ab', 0, bottom=i, edgecolor='k', fill=False)
            else:
                ab_bias = followed_bias_ab[filename]
                ax.bar(exp_day + '_ab', 1, bottom=i, color=colors_bias[ab_bias], label=ab_bias)
            ax.text(exp_day + '_ab', i + 0.1, filename.split('_')[2][-2:])

            if filename not in followed_bias_c:
                ax.bar(exp_day + '_c', 0, bottom=i, edgecolor='k', fill=False)
            else:
                c_bias = followed_bias_c[filename]
                ax.bar(exp_day + '_c', 1, bottom=i, color=colors_bias[c_bias], label=c_bias, alpha=0.5)
            # ax.text(exp_day + '_c', i, filename.split('_')[2][-2:])

            if filename not in followed_bias_f:
                ax.bar(exp_day + '_f', 0, bottom=i, edgecolor='k', fill=False)
            else:
                f_bias = followed_bias_f[filename]
                ax.bar(exp_day + '_f', 1, bottom=i, color=colors_bias[f_bias], label=f_bias, alpha=0.5)
            # ax.text(exp_day + '_f', i, filename.split('_')[2][-2:])
        reduce_legend(ax)

    # plot vertical dotted lines between every _c and _ab label
    for i in range(len(exp_days(size))):
        ax.axvline(2 + 3 * i + 0.5, color='k', ls='--', lw=0.5)

    # right = sum([bias == 'right' for bias in biases])
    # left = sum([bias == 'left' for bias in biases])
    # none = sum([bias is None for bias in biases])
    #
    # ax.bar(exp_day, right, color='y', label='right')
    # ax.bar(exp_day, left, bottom=right, color='b', label='left')
    # ax.bar(exp_day, none, bottom=right + left, color='r', label='no bias')
    # ax.legend()

    # if right > 0:
    #     ax.text(exp_day, 1 / 2 * right, 'right: ' + str(right), ha='center', va='center', color='k')
    # if left > 0:
    #     ax.text(exp_day, right + 1/2 * left, 'left: ' + str(left), ha='center', va='center', color='w')
    # if none > 0:
    #     ax.text(exp_day, right + left + 1 / 2 * none, 'no bias: ' + str(none), ha='center', va='center', color='k')


def calc_biases_ab(df, exp_day, delta_t=20):
    traj_all = None
    followed_bias = {}
    for filename in tqdm(df['filename'], desc='calculating biases ' + size + ' ' + str(exp_day)):
        traj = get(filename)
        ts = extend_time_series_to_match_frames(time_series_dict[filename], len(traj.frames))
        if 'b' in ts and 'c' in ts:
            last = len(ts) - ts[::-1].index('b')
            if 'c' in ts[last:]:
                first = ts[last:].index('c') + last
                frames = [max(0, last - delta_t * traj.fps),
                          min(len(traj.frames), first + delta_t * traj.fps)]
                x_sub = traj.cut_off(frame_indices=frames)
                followed_bias[filename] = traj_followed_bias_flipping_ab(x_sub)
                # if traj_all is None:
                #     traj_all = x_sub
                # else:
                #     traj_all = traj_all + x_sub
    # traj_all.play(step=10, bias=bias_direction[exp_day],
    #               # videowriter=1
    #               )
    return followed_bias


def calc_biases_c(df, exp_day, delta_t=20):
    # traj_all = None
    followed_bias = {}
    for filename in tqdm(df['filename'], desc='calculating biases ' + size + ' ' + str(exp_day) + 's'):
        # filename = 'XL_SPT_4640007_XLSpecialT_1_ants (part 1)'
        traj = get(filename)
        ts = extend_time_series_to_match_frames(time_series_dict[filename], len(traj.frames))
        if 'c' in ts and 'e' in ts:
            last = len(ts) - ts[::-1].index('c')
            if 'e' in ts[last:]:
                first = ts[last:].index('e') + last
                frames = [max(0, last - delta_t * traj.fps),
                          min(len(traj.frames), first + delta_t * traj.fps)]
                x_sub = traj.cut_off(frame_indices=frames)
                followed_bias[filename] = traj_followed_bias_flipping_c(x_sub)
                # if traj_all is None:
                #     traj_all = x_sub
                # else:
                #     traj_all = traj_all + x_sub
    # traj_all.play(step=10,
    #               # bias=bias_direction[exp_day],
    #               videowriter=1
    #               )
    return followed_bias


def calc_biases_f(df, exp_day, delta_t=20):
    # traj_all = None
    followed_bias = {}
    for filename in tqdm(df['filename'], desc='calculating biases ' + size + ' ' + str(exp_day)):
        # filename = 'XL_SPT_4640007_XLSpecialT_1_ants (part 1)'
        traj = get(filename)
        ts = extend_time_series_to_match_frames(time_series_dict[filename], len(traj.frames))
        if 'f' in ts and 'h' in ts:
            last = len(ts) - ts[::-1].index('c')
            if 'h' in ts[last:]:
                first = ts[last:].index('h') + last
                frames = [max(0, last - delta_t * traj.fps),
                          min(len(traj.frames), first + delta_t * traj.fps)]
                x_sub = traj.cut_off(frame_indices=frames)
                followed_bias[filename] = traj_followed_bias_flipping_f(x_sub)
                # if traj_all is None:
                #     traj_all = x_sub
                # else:
                #     traj_all = traj_all + x_sub
    # traj_all.play(step=10,
    #               # bias=bias_direction[exp_day],
    #               videowriter=1
    #               )
    return followed_bias


def exp_days(size):
    df_ant_size = df_ant_all[df_ant_all['size'] == size]
    df_ant_size = df_ant_size.sort_values(by='filename')
    df_ant_size.loc[:, 'exp_day'] = df_ant_size['filename'].apply(find_exp_day)
    df_ant_size = df_ant_size.groupby('exp_day')
    return dict(list(df_ant_size)).items()


if __name__ == '__main__':
    # # ++++++ AB turning bias ++++++
    #
    # followed_bias_all_ab = {}
    # for size in ['S (> 1)', 'M', 'L', 'XL', ]:
    #     # save followed_bias
    #     for exp_day, df_ant_day in exp_days(size):
    #         followed_bias_all_ab.update(calc_biases_ab(df_ant_day, exp_day))
    # with open('followed_bias_ab.json', 'w') as f:
    #     json.dump(followed_bias_all_ab, f)

    # # ++++++ C turning bias ++++++
    #
    # followed_bias_all_c = {}
    # for size in ['XL', 'S (> 1)', 'M', 'L', ]:
    #     # save followed_bias
    #     for exp_day, df_ant_day in exp_days(size):
    #         followed_bias_all_c.update(calc_biases_c(df_ant_day, exp_day))
    # with open('followed_bias_c.json', 'w') as f:
    #     json.dump(followed_bias_all_c, f)

    # ++++++ F turning bias ++++++

    # followed_bias_all_f = {}
    # for size in ['XL', 'M', 'S (> 1)', 'L', ]:
    #     # save followed_bias
    #     for exp_day, df_ant_day in exp_days(size):
    #         followed_bias_all_f.update(calc_biases_f(df_ant_day, exp_day))
    # with open('followed_bias_f.json', 'w') as f:
    #     json.dump(followed_bias_all_f, f)

    # ++++++ PLOTTING ++++++

    with open('followed_bias_ab.json', 'r') as f:
        followed_bias_all_ab = json.load(f)

    with open('followed_bias_c.json', 'r') as f:
        followed_bias_all_c = json.load(f)

    with open('followed_bias_f.json', 'r') as f:
        followed_bias_all_f = json.load(f)

    fig, axs = plt.subplots(4, 1, figsize=(15, 12))
    for size, ax in zip(['S (> 1)', 'M', 'L', 'XL', ], axs):
        # save followed_bias
        plot_biases(followed_bias_all_ab, followed_bias_all_c, followed_bias_all_f, ax, size)
        ax.set_ylabel('bias direction')
        ax.set_xlabel('exp_day')
        ax.set_ylim([0, 16])
        ax.set_xlim([-1, 36])
        # make title bold and big
        ax.set_title(size, fontweight="bold", size=20)
    plt.tight_layout()
    plt.savefig('followed_bias.png')
    plt.savefig('followed_bias.svg', transparent=True)
    plt.savefig('followed_bias.pdf', transparent=True)
    DEBUG = 1
