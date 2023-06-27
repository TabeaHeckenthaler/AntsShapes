import numpy as np
import matplotlib.pyplot as plt
from DataFrame.import_excel_dfs import df_ant_excluded
from os import path
from Directories import home, network_dir
import json
from itertools import groupby

with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

save_names = {'S (> 1)': 'SmallMore', 'Single (1)': 'Single', 'XL': 'XL',
              'L': 'L', 'M': 'M'}

sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)', ]


def time_stamped_series(time_series, time_step) -> list:
    groups = groupby(time_series)
    return [(label, sum(1 for _ in group) * time_step) for label, group in groups]


# from all the time_series change the following states
original_states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h', 'i']
# to the following states
goal_states = ['ab', 'ac', 'ab', 'ab', 'ab', 'ab', 'c', 'c', 'e', 'e', 'e', 'f', 'h', 'h', 'i']

transform = dict(zip(original_states, goal_states))
u = np.unique(goal_states).tolist()
passages_forwards = [str(st1) + ' -> ' + str(st2) for st1, st2 in zip(u[:-1], u[1:])]
passages_backwards = [str(st2) + ' -> ' + str(st1) for st1, st2 in zip(u[:-1], u[1:])]
passages = passages_forwards + passages_backwards


def save_dictionaries():
    for size, df in df_ant_excluded.groupby('size'):
        passage_times = {passage: [] for passage in passages}
        for filename in df['filename']:
            time_series = [transform[state] for state in time_series_dict[filename]]
            tss = time_stamped_series(time_series, 0.25)

            # go through all the passages and find the time it took to pass
            for st1, st2 in zip(tss[:-1], tss[1:]):
                passage_times[str(st1[0]) + ' -> ' + str(st2[0])].append((filename, st1[1]))

        # save the passage times
        with open(path.join(home, 'Analysis', 'source_of_scaling', 'passage_times', f'{save_names[size]}.json'),
                  'w') as json_file:
            json.dump(passage_times, json_file)
            json_file.close()

        # for every filename, find the time_series
        DEBUG = 1


def scale_dictionary(passage_times, size):
    scale = {'Single (1)': 8, 'S (> 1)': 8, 'M': 4, 'L': 2, 'XL': 1}
    for passage in passages:
        passage_times[passage] = [(filename, time * scale[size])
                                  for filename, time in passage_times[passage]]
    return passage_times


def plot_mean_passage_times(passage_times, size, name, ax, ax2):
    means, std = {}, {}
    for passage in passages:
        means[passage] = np.median([time for _, time in passage_times[passage]])
        std[passage] = np.std([time for _, time in passage_times[passage]]) / np.sqrt(len(passage_times[passage]))

    # ax.errorbar([p[:2] for p in passages_forwards], [means[m] for m in passages_forwards],
    ax.errorbar([pos + 0.5 for pos in range(6)], [means[m] for m in passages_forwards],
                yerr=[std[m] for m in passages_forwards], capsize=5, label=size + ' forwards',
                color=colors[size])
    # make a twin axis
    ax2.errorbar([pos + 0.5 for pos in range(7)], [np.nan] + [means[m] for m in passages_backwards],
                 yerr=[np.nan] + [std[m] for m in passages_backwards], capsize=5, color=colors[size],
                 linestyle='--', label=size + ' backwards')

    # set the xticks
    ax.set_xticks([pos for pos in range(7)])
    ax.set_xticklabels(['ab', 'ac', 'c', 'e', 'f', 'h', 'i'])

    # have legends of axis ax and ax2 in the same legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')


def plot_box_passage_times(passage_times, size, name, ax, ax2):
    # plot the passage times
    # plot for all sizes
    boxes = []
    for passage in passages_forwards:
        boxes.append([time for _, time in passage_times[passage]])
    bp = ax.boxplot(boxes, positions=[pos + 0.3 for pos in range(6)], widths=0.2, patch_artist=True)

    boxes = []
    for passage in passages_backwards:
        boxes.append([time for _, time in passage_times[passage]])
    bp1 = ax.boxplot(boxes, positions=[pos + 0.7 for pos in range(6)], widths=0.2, patch_artist=True)
    # set boxplot face color bp1 as red
    for element in ['boxes']:
        plt.setp(bp1[element], color='red')

    for element in ['medians']:
        plt.setp(bp1[element], color='black')

    # make ax2 red
    ax2.spines['right'].set_color('red')

    # set the xticks
    ax.set_xticks([pos for pos in range(7)])
    ax.set_xticklabels(['ab', 'ac', 'c', 'e', 'f', 'h', 'i'])

    # have legends of axis ax and ax2 in the same legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')


if __name__ == '__main__':
    # save_dictionaries()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)', ][:-1]
    colors = {'XL': 'blue', 'L': 'green', 'M': 'orange', 'S (> 1)': 'black', 'Single (1)': 'red', }
    for size in sizes:
        name = save_names[size]
        with open(path.join(home, 'Analysis', 'source_of_scaling', 'passage_times', f'{name}.json'), 'r') as json_file:
            passage_times = json.load(json_file)
            json_file.close()

        passage_times = scale_dictionary(passage_times, size)
        plot_mean_passage_times(passage_times, size, name, ax, ax2)
    ax.set_ylim(0, 1000)
    ax2.set_ylim(0, 1000)
    ax.set_ylabel('time [s] (forwards)')
    ax2.set_ylabel('time [s] (backwards)')
    ax.set_xlabel('size')
    fig.savefig(path.join(home, 'Analysis', 'source_of_scaling', 'passage_times', 'median.png'))

    # sizes = ['XL', 'L', 'M', 'S (> 1)', 'Single (1)', ][:-1]
    # colors = {'XL': 'blue', 'L': 'green', 'M': 'orange', 'S (> 1)': 'black', 'Single (1)': 'red', }
    # for size in sizes:
    #     fig, ax = plt.subplots()
    #     ax2 = ax.twinx()
    #     name = save_names[size]
    #     with open(path.join(home, 'Analysis', 'source_of_scaling', 'passage_times', f'{name}.json'), 'r') as json_file:
    #         passage_times = json.load(json_file)
    #         json_file.close()
    #
    #     passage_times = scale_dictionary(passage_times, size)
    #     plot_box_passage_times(passage_times, size, name, ax, ax2)
    #     ax.set_ylim(0, 800)
    #     ax2.set_ylim(0, 800)
    #     ax.set_ylabel('time [s] (forwards)')
    #     ax2.set_ylabel('time [s] (backwards)')
    #     ax.set_xlabel('size')
    #     fig.savefig(path.join(home, 'Analysis', 'source_of_scaling', 'passage_times', name + '_box_plot.png'))

    DEBUG = 1