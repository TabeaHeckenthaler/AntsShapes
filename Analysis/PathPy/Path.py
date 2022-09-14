from itertools import groupby
import numpy as np
from trajectory_inheritance.get import get
from trajectory_inheritance.exp_types import solver_geometry
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled, pre_final_state
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_AdditionalStates
from Analysis.PathLength.PathLength import PathLength
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from tqdm import tqdm
from DataFrame.dataFrame import myDataFrame
from Directories import network_dir
import os
import json
from matplotlib import pyplot as plt


time_step = 0.25  # seconds


class Path:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """

    def __init__(self, time_step: float, time_series=None, x=None, conf_space_labeled=None, only_states=False):
        """
        :param step: after how many frames I add a label to my label list
        :param x: trajectory
        :return: list of strings with labels
        """
        self.time_step = time_step
        self.time_series = time_series
        if self.time_series is None and x is not None:
            self.time_series = self.get_time_series(conf_space_labeled, x)

            # # for 'small_20220308115942_20220308120334'
            # if self.time_series[114:125] == ['bd', 'bd', 'bd', 'db', 'db', 'd', 'db', 'db', 'd', 'ba', 'ba']:
            #     self.time_series[114:125] = ['bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'ba', 'ba', 'ba', 'ba']
            # # x.play(path=self, wait=15)

            self.time_series = conf_space_labeled.correct_time_series(self.time_series, filename=x.filename)
            # self.save_transition_images(x)
            if only_states:
                self.time_series = [l[0] for l in self.time_series]
        self.state_series = self.calculate_state_series(self.time_series)

    @staticmethod
    def show_configuration(x, coords, save_dir='', text_in_snapshot='', frame=0):
        maze = Maze(x)
        display = Display(text_in_snapshot, 1, maze, frame=frame)
        maze.set_configuration(coords[0:2], coords[2])
        maze.draw(display)
        display.display()
        if len(save_dir) > 0:
            display.snapshot(save_dir)

    def save_transition_images(self, x):
        coords = [coords for coords in x.iterate_coords(step=self.time_step)]
        Path.show_configuration(x, coords[0],
                                'State_transition_images\\' + str(int(0 * self.time_step)) + '.png',
                                text_in_snapshot=self.time_series[0], frame=0)
        for i, (label0, label1) in enumerate(zip(self.time_series[:-1], self.time_series[1:])):
            if label0 != label1:
                Path.show_configuration(x, coords[i],
                                        'State_transition_images\\' + str(int(i * self.time_step)) + '.png',
                                        text_in_snapshot=label1, frame=int(i * self.time_step * x.fps))

    def get_time_series(self, conf_space_labeled, x):
        coords = [coords for coords in x.iterate_coords(time_step=self.time_step)]
        indices = [conf_space_labeled.coords_to_indices(*coords) for coords in coords]
        labels = [None]
        for i, index in enumerate(indices):
            # if i == 11360:
            #     DEBUG = 1
            labels.append(self.label_configuration(index, conf_space_labeled, last_label=labels[-1]))
        labels = labels[1:]

        if pre_final_state in labels[-1] and pre_final_state != labels[-1]:
            labels.append(pre_final_state)

        return labels

    def label_configuration(self, index, conf_space_labeled, last_label=None) -> str:
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            label = conf_space_labeled.find_closest_state(index, last_label=last_label)
        # if set(label) == set('d'):
        #     conf_space_labeled.draw_ind(index)
        return label

    @staticmethod
    def combine_transitions(labels) -> list:
        """
        I want to combine states, that are [.... 'gb' 'bg'...] to [... 'gb'...]
        """
        labels = [''.join(sorted(state)) for state in labels]
        mask = [True] + [sorted(state1) != sorted(state2) for state1, state2 in zip(labels, labels[1:])]
        return np.array(labels)[mask].tolist()

    @staticmethod
    def symmetrize(state_series):
        return [state.replace('d', 'e') for state in state_series]

    @staticmethod
    def only_states(state_series):
        return [state[0] for state in state_series]

    def state_at_time(self, time: float) -> str:
        return self.time_series[int(time / self.time_step)]

    # def interpolate_zeros(self, labels: list) -> list:
    #     """
    #     Interpolate over all the states, that are not inside Configuration space (due to the computer representation of
    #     the maze not being exactly the same as the real maze)
    #     :return:
    #     """
    #
    #     if labels[0] == '0':
    #         labels[0] = [l for l in labels if l != '0'][:1000][0]
    #
    #     for i, l in enumerate(labels):
    #         if l == '0':
    #
    #             where = np.where(np.array(labels[i:]) != '0')[0]
    #
    #             if len(where) < 1:
    #                 if labels[i - 1] == final_state + pre_final_state:
    #                     labels[i:] = [final_state for _ in range(len(labels[i:]))]
    #                 else:
    #                     labels[i:] = [labels[i - 1] for _ in range(len(labels[i:]))]
    #
    #             else:
    #                 index = where[0] + i
    #                 if self.valid_transition(labels[i - 1], labels[index]):
    #                     transitions = [labels[i - 1], labels[index]]
    #                 else:
    #                     if labels[i - 1] == 'e' and labels[index] == 'g':
    #                         transitions = ['e', 'e']  # this occurs only in small SPT ants
    #                     else:
    #                         transitions = [labels[i - 1], *self.necessary_transitions(labels[i - 1], labels[index]),
    #                                        labels[index]]
    #
    #                 for array, state in zip(np.array_split(range(i, index), len(transitions)), transitions):
    #                     if len(array > 0):
    #                         labels[array[0]: array[-1] + 1] = [state for _ in range(len(array))]
    #
    #     if len([1 for state1, state2 in zip(labels[:-1], labels[1:]) if
    #             state1 == final_state and state2 != final_state]):
    #         print('something')
    #
    #     return labels

    @staticmethod
    def calculate_state_series(time_series):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series])]
        labels = Path.combine_transitions(labels)
        labels = Path.add_final_state(labels)
        return labels

    @staticmethod
    def time_stamped_series(time_series, time_step) -> list:
        groups = groupby(time_series)
        return [(label, sum(1 for _ in group) * time_step) for label, group in groups]

    @staticmethod
    def path_length_stamped_series(filename, time_series, time_step) -> list:
        x = get(filename)
        hey = Path.time_stamped_series(time_series, time_step)
        frames = [(h[0], int(h[1] * x.fps)) for h in hey]
        hey_path_length = []
        current = 0

        for (state, frame_number) in frames:
            new_end_frame = min(current + frame_number, len(x.frames))
            path_l = PathLength(x).calculate_path_length(frames=[current, new_end_frame])
            current = new_end_frame
            hey_path_length.append((state, path_l))
        # x.play()
        return hey_path_length

    @classmethod
    def create_dicts(cls, myDataFrame, ConfigSpace_class=ConfigSpace_Labeled):
        dictio_ts = {}
        dictio_ss = {}
        shape = 'SPT'
        myDataFrame = myDataFrame[myDataFrame['shape'] == shape]

        for solver in myDataFrame['solver'].unique():
            print(solver)
            df = myDataFrame[myDataFrame['solver'] == solver].sort_values('size')
            groups = df.groupby(by=['size'])
            for size, cs_group in groups:
                if size == 'L':
                    print('only L')
                    cs_labeled = ConfigSpace_class(solver, size, shape, solver_geometry[solver])
                    cs_labeled.load_labeled_space()
                    for _, exp in tqdm(cs_group.iterrows()):
                        print(exp['filename'])
                        if (exp['maze dimensions'], exp['load dimensions']) != solver_geometry[solver]:
                            dictio_ts[exp['filename']] = None
                            dictio_ss[exp['filename']] = None
                        else:
                            x = get(exp['filename'])
                            path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                            dictio_ts[exp['filename']] = path_x.time_series
                            dictio_ss[exp['filename']] = path_x.state_series
        return dictio_ts, dictio_ss

    @classmethod
    def add_to_dict(cls, myDataFrame, ConfigSpace_class, time_series_dict, state_series_dict, solver='ant') -> tuple:
        """

        """
        dictio_ts = {}
        dictio_ss = {}
        print('only ants')
        myDataFrame = myDataFrame[myDataFrame['shape'] == 'SPT'][myDataFrame['solver'] == solver]
        to_add = myDataFrame[~myDataFrame['filename'].isin(time_series_dict.keys())]
        size_groups = to_add.groupby('size')

        for size, cs_group in size_groups:
            print(size)
            cs_labeled = ConfigSpace_class(solver, size, 'SPT', solver_geometry[solver])
            cs_labeled.load_labeled_space()
            for _, exp in tqdm(cs_group.iterrows()):
                print(exp['filename'])
                if (exp['maze dimensions'], exp['load dimensions']) != solver_geometry[solver]:
                    dictio_ts[exp['filename']] = None
                    dictio_ss[exp['filename']] = None
                else:
                    x = get(exp['filename'])
                    path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                    dictio_ts[exp['filename']] = path_x.time_series
                    dictio_ss[exp['filename']] = path_x.state_series
        time_series_dict.update(dictio_ts)
        state_series_dict.update(dictio_ss)
        return time_series_dict, state_series_dict

    def bar_chart(self, ax, axis_label='', winner=False, food=False, block=False):
        ts = self.time_series
        # ts = Path.only_states(self.time_series)
        # ts = Path.symmetrize(ts)
        # dur = Path.state_duration(ts)
        dur = Path.time_stamped_series(ts, self.time_step)

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # color_dict = {state: color for state, color in zip(['a', 'b', 'c', 'd', 'h', 'f', 'i'], colors)}

        color_dict = {'a': '#1f77b4', 'b': '#ff7f0e', 'c': '#2ca02c', 'e': '#d62728', 'f': '#8c564b', 'h': '#9467bd',
                      'i': '#e377c2',
                      'a1': '', 'a2': '', 'be': '', 'bf': '', 'eb': '', 'eg': ''}
        left = 0

        given_names = {}

        for name, duration in dur:
            dur_in_min = duration / 60
            if block:
                b = ax.barh(axis_label, 1, color=color_dict[name], left=left, label=name)
                left += 1
            else:
                b = ax.barh(axis_label, dur_in_min, color=color_dict[name], left=left, label=name)
                left += dur_in_min
            if name not in given_names:
                given_names.update({name: b})

        # if winner:
        #     plt.text(left + 1, b.patches[0].xy[-1], 'v', color='green')
        # else:
        #     plt.text(left + 1, b.patches[0].xy[-1], 'x', color='red')
        #
        # if food:
        #     plt.text(left + 2, b.patches[0].xy[-1], 'f', color='black')

        labels = list(color_dict.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in labels]
        plt.legend(handles, labels)

    @staticmethod
    def get_dicts(name=''):
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'r') as json_file:
            time_series_dict = json.load(json_file)
            json_file.close()

        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'r') as json_file:
            state_series_dict = json.load(json_file)
            json_file.close()
        return time_series_dict, state_series_dict

    @staticmethod
    def save_dicts(time_series_dict, state_series_dict, name=''):
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'w') as json_file:
            json.dump(time_series_dict, json_file)
            json_file.close()

        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'w') as json_file:
            json.dump(state_series_dict, json_file)
            json_file.close()


time_series_dict, state_series_dict = Path.get_dicts()
time_series_dict_selected_states, state_series_dict_selected_states = Path.get_dicts(name='_selected_states')

DEBUG = 1
if __name__ == '__main__':
    # filename = 'S_SPT_4750016_SSpecialT_1_ants (part 1)'
    # x = get(filename)
    # x.play()
    # cs_labeled = ConfigSpace_AdditionalStates(x.solver, x.size, x.shape, x.geometry())
    # cs_labeled.load_labeled_space()
    # path = Path(time_step, x=x, conf_space_labeled=cs_labeled)
    # print(path.state_series)
    # x.play(path=path)

    ConfigSpace_class = ConfigSpace_AdditionalStates
    time_series_dict, state_series_dict = Path.create_dicts(myDataFrame, ConfigSpace_class)
    # time_series_dict, state_series_dict = Path.add_to_dict(myDataFrame, ConfigSpace_class, time_series_dict, state_series_dict)

    Path.save_dicts(time_series_dict, state_series_dict, name='_selected_states')
