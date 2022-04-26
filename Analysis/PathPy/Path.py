from itertools import groupby
import numpy as np
from trajectory_inheritance.trajectory import get
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
from Analysis.PathPy.SPT_states import final_state, allowed_transition_attempts
from Analysis.PathLength import PathLength
from Setup.Maze import Maze
from PhysicsEngine.Display import Display


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

        if x is not None:
            self.frame_step = int(self.time_step * x.fps)  # in seconds
        else:
            self.frame_step = None

        self.time_series = time_series
        if self.frame_step is not None and self.time_series is None and x is not None:
            self.time_series = self.get_time_series(conf_space_labeled, x, only_states=only_states)
            self.save_transition_images(x)

            # for 'small_20220308115942_20220308120334'
            if self.time_series[114:125] == ['bd', 'bd', 'bd', 'db', 'db', 'd', 'db', 'db', 'd', 'ba', 'ba']:
                self.time_series[114:125] = ['bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'ba', 'ba', 'ba', 'ba']
            # x.play(path=self, wait=15)

            self.correct_time_series()

        self.state_series = self.calculate_state_series()

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
        coords = [coords for coords in x.iterate_coords(step=self.frame_step)]
        Path.show_configuration(x, coords[0],
                                'State_transition_images\\' + str(int(0 * self.time_step)) + '.png',
                                text_in_snapshot=self.time_series[0], frame=0)
        for i, (label0, label1) in enumerate(zip(self.time_series[:-1], self.time_series[1:])):
            if label0 != label1:
                Path.show_configuration(x, coords[i],
                                        'State_transition_images\\' + str(int(i * self.time_step)) + '.png',
                                        text_in_snapshot=label1, frame=int(i * self.time_step * x.fps))

    def get_time_series(self, conf_space_labeled, x, only_states=False):
        print(x)
        coords = [coords for coords in x.iterate_coords(step=self.frame_step)]
        indices = [conf_space_labeled.coords_to_indices(*coords) for coords in coords]
        labels = [None]
        for i, index in enumerate(indices):
            labels.append(self.label_configuration(index, conf_space_labeled, last_label=labels[-1]))
        if only_states:
            labels = [l[0] for l in labels]
        return labels[1:]

    def correct_time_series(self):
        self.time_series = self.cut_off_after_final_state(self.time_series)
        self.time_series = self.delete_false_transitions(self.time_series)
        self.time_series = self.get_rid_of_short_lived_states(self.time_series)
        self.time_series = self.add_missing_transitions(self.time_series)

    def label_configuration(self, index, conf_space_labeled, last_label=None) -> str:
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            label = conf_space_labeled.find_closest_state(index, last_label=last_label)
        # if set(label) == set('d'):
        #     conf_space_labeled.draw_ind(index)
        return label

    def get_rid_of_short_lived_states(self, labels, min=5):
        grouped = [(''.join(k), sum(1 for _ in g)) for k, g in groupby([tuple(label) for label in labels])]
        new_labels = [grouped[0][0] for _ in range(grouped[0][1])]
        for i, (label, length) in enumerate(grouped[1:-1], 1):
            if length <= min and self.valid_transition(new_labels[-1], grouped[i + 1][0]):
                print(grouped[i - 1][0] + ' => ' + grouped[i + 1][0])
                new_labels = new_labels + [new_labels[-1] for _ in range(length)]
            else:
                new_labels = new_labels + [label for _ in range(length)]
        new_labels = new_labels + [grouped[-1][0] for _ in range(grouped[-1][1])]
        return new_labels

    @staticmethod
    def valid_state_transition(state1, state2):
        if len(state1) == 2 and state1[::-1] == state2:  # is transition like ab -> ba
            return state1 in allowed_transition_attempts
        else:
            return True

    @staticmethod
    def delete_false_transitions(labels):
        new_labels = [labels[0]]
        for ii, next_state in enumerate(labels[1:], start=1):
            if not Path.valid_state_transition(new_labels[-1], next_state):
                new_labels.append(new_labels[-1])
            else:
                new_labels.append(next_state)
        return new_labels

    @staticmethod
    def cut_off_after_final_state(labels):
        first_time_in_final_state = np.where(np.array(labels) == final_state)[0]
        if len(first_time_in_final_state) > 1:
            # print(labels[:first_time_in_final_state[0] + 1][-10:])
            return labels[:first_time_in_final_state[0] + 1]
        else:
            return labels

    @staticmethod
    def combine_transitions(labels) -> list:
        """
        I want to combine states, that are [.... 'gb' 'bg'...] to [... 'gb'...]
        """
        labels = [''.join(sorted(state)) for state in labels]
        mask = [True] + [sorted(state1) != sorted(state2) for state1, state2 in zip(labels, labels[1:])]
        return np.array(labels)[mask].tolist()

    @staticmethod
    def valid_transition(state1, state2):
        if not Path.valid_state_transition(state1, state2):
            return False
        if set(state1) == set(state2):
            return True
        elif set(state1) in [set('fg'), set('fd')] and set(state2) in [set('fg'), set('fd')]:
            return False
        return state1[0] == state2[0] or len(set(state1) & set(state2)) == 2

    @staticmethod
    def necessary_transitions(state1, state2, ii: int = '', frame_step=1) -> list:
        if state1 == 'c' and state2 == 'fh':
            return ['ce', 'e', 'ef', 'f']
        if state1 == 'c' and state2 == 'f':
            return ['ce', 'e', 'ef']
        if state1 == 'ba' and state2 == 'cg':
            return ['a', 'ac', 'c']

        # otherwise, our Markov chain is not absorbing for L ants
        if set(state1) in [set('ef'), set('ec')] and set(state1) in [set('ef'), set('ec')]:
            return ['e']

        if len(state1) == len(state2) == 1:
            transition = ''.join(sorted(state1 + state2))
            if transition in allowed_transition_attempts:
                return [transition]
            else:
                print('Skipped 3 states: ' + state1 + ' -> ' + state2 + ' in ii ' + str(ii))
                return []

        elif len(state1) == len(state2) == 2:
            print('Moved from transition to transition: ' + state1 + '_' + state2 + ' in ii ' + str(ii))
            return []

        elif ''.join(sorted(state1 + state2[0])) in allowed_transition_attempts:
            return [''.join(sorted(state1 + state2[0])), state2[0]]
        elif ''.join(sorted(state1[0] + state2)) in allowed_transition_attempts:
            return [state1[0], ''.join(sorted(state1[0] + state2))]

        elif len(state2) > 1 and ''.join(sorted(state1 + state2[1])) in allowed_transition_attempts:
            return [''.join(sorted(state1 + state2[1])), state2[1]]
        elif len(state1) > 1 and ''.join(sorted(state1[1] + state2)) in allowed_transition_attempts:
            return [state1[1], ''.join(sorted(state1[1] + state2))]
        else:
            print('What happened: ' + state1 + ' -> ' + state2 + ' in ii ' + str(ii))
            return []

    def add_missing_transitions(self, labels) -> list:
        """
        I want to correct states series, that are [.... 'g' 'b'...] to [... 'g' 'gb' 'b'...]
        """
        new_labels = [labels[0]]

        for ii, state2 in enumerate(labels[1:]):
            # if state1 in ['cg', 'ac'] and state2 in ['cg', 'ac'] and state1 != state2:
            #     DEBUG = 1
            state1 = new_labels[-1]
            if not self.valid_transition(state1, state2):
                if state1 in ['f', 'e'] and state2 == 'i':
                    new_labels.append(state1)  # only for small SPT ants
                elif state1 in ['eg', 'dg', 'cg'] and state2 == 'g':
                    new_labels.append(state1)  # only for small SPT ants
                elif state1 == 'ba' and state2 in ['d', 'e']:
                    new_labels.append(state1)
                elif len(state2) == 2 and state1 == state2[1]:
                    new_labels.append(state2[1] + state2[0])
                else:
                    for t in self.necessary_transitions(state1, state2, ii=ii, frame_step=self.frame_step):
                        new_labels.append(t)
                    new_labels.append(state2)
            else:
                new_labels.append(state2)
        return new_labels

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

    def calculate_state_series(self):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        labels = self.combine_transitions(labels)
        labels = self.cut_off_after_final_state(labels)
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
        # TODO: 'XL_SPT_4640014_XLSpecialT_1_ants' for example is a bit off...
        # x.play()
        return hey_path_length


if __name__ == '__main__':
    # filenames = ['large_20210805171741_20210805172610_perfect',
    #             'medium_20210507225832_20210507230303_perfect',
    #             'small2_20220308120548_20220308120613_perfect']
    # filenames = ['small_20220308115942_20220308120334']

    filenames = ['XL_SPT_4640014_XLSpecialT_1_ants']
    for filename in filenames:
        time_step = 0.25  # seconds
        x = get(filename)
        cs_labeled = ConfigSpace_Labeled(x.solver, x.size, x.shape, x.geometry())
        cs_labeled.load_labeled_space()
        # cs_labeled.visualize_space(space=cs_labeled.space_labeled == 'a')
        # x.play(cs=cs_labeled, frames=[24468 - 1000, 24468 + 100], step=1)

        path = Path(time_step, x=x, conf_space_labeled=cs_labeled)
        print(path.time_series)
        DEBUG = 1
