from itertools import groupby
import numpy as np
from trajectory_inheritance.trajectory import get
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled
from Analysis.PathPy.SPT_states import pre_final_state, final_state, allowed_transition_attempts


class Path:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """

    def __init__(self, time_step: float, time_series=None, x=None, conf_space_labeled=None):
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
            self.time_series = self.get_time_series(conf_space_labeled, x)

        self.state_series = self.calculate_state_series()

    def get_time_series(self, conf_space_labeled, x):
        indices = [conf_space_labeled.coords_to_indices(*coords) for coords in x.iterate_coords(step=self.frame_step)]
        labels = [self.label_configuration(index, conf_space_labeled) for index in indices]
        labels = self.cut_off_after_final_state(labels)
        # labels = self.interpolate_zeros(labels)  # TODO: it would be better to find the closest non-zero state.
        labels = self.delete_false_transitions(labels)
        labels = self.get_rid_of_short_lived_states(labels)
        labels = self.add_missing_transitions(labels)
        return labels

    def label_configuration(self, index, conf_space_labeled) -> str:
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            label = conf_space_labeled.find_closest_state(index)
        return label

    def get_rid_of_short_lived_states(self, labels, min=5):
        grouped = [(''.join(k), sum(1 for _ in g)) for k, g in groupby([tuple(label) for label in labels])]
        new_labels = [grouped[0][0] for _ in range(grouped[0][1])]
        for i, (label, length) in enumerate(grouped[1:-1], 1):
            if length <= min and self.valid_transition(grouped[i - 1][0], grouped[i + 1][0]):
                new_labels = new_labels + [new_labels[-1] for _ in range(length)]
            else:
                new_labels = new_labels + [label for _ in range(length)]
        new_labels = new_labels + [grouped[-1][0] for _ in range(grouped[-1][1])]
        return new_labels

    @staticmethod
    def delete_false_transitions(labels):
        not_allowed = [('bg', 'gb')]
        last_state = labels[0]
        labels_copy = labels.copy()
        for ii, next_state in enumerate(labels[1:], start=1):
            # if (last_state, next_state) == ('bg', 'gb'):
            #     DEBUG = 1
            if (last_state, next_state) in not_allowed:
                labels_copy[ii] = last_state
            last_state = labels_copy[ii]
        return labels_copy

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
        if set(state1) == set(state2):
            return True
        elif set(state1) in [set('fg'), set('fd')] and set(state2) in [set('fg'), set('fd')]:
            return False
        return len(set(state1) & set(state2)) > 0

    @staticmethod
    def neccessary_transitions(state1, state2) -> list:
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
                raise ValueError('Skipped 3 states: ' + state1 + ' -> ' + state2)

        elif len(state1) == len(state2) == 2:
            print('Moved from transition to transition: ' + state1 + '_' + state2)
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
            raise ValueError('What happened: ' + state1 + ' -> ' + state2)

    def add_missing_transitions(self, labels) -> list:
        """
        I want to correct states series, that are [.... 'g' 'b'...] to [... 'g' 'gb' 'b'...]
        """
        labels_copy = labels.copy()
        i = 1

        for ii, (state1, state2) in enumerate(zip(labels[:-1], labels[1:])):
            if not self.valid_transition(state1, state2):
                if state1 in ['f', 'e'] and state2 == 'i':
                    labels_copy[ii + i] = state1  # only for small SPT ants
                else:
                    for t in self.neccessary_transitions(state1, state2):
                        labels_copy.insert(ii + i, t)
                        i += 1

        return labels_copy

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
    #                         transitions = [labels[i - 1], *self.neccessary_transitions(labels[i - 1], labels[index]),
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

    def time_stamped_series(self) -> list:
        groups = groupby(self.time_series)
        return [(label, sum(1 for _ in group) * self.time_step) for label, group in groups]


if __name__ == '__main__':
    filename = 'M_SPT_4700005_MSpecialT_1_ants'
    x = get(filename)
    cs_labeled = ConfigSpace_Labeled(x.solver, x.size, x.shape, x.geometry())
    cs_labeled.load_labeled_space()
    # cs_labeled.visualize_space()
    # x.play(cs=cs_labeled, step=5)

    path = Path(0.25, x=x, conf_space_labeled=cs_labeled)
    DEBUG = 1
