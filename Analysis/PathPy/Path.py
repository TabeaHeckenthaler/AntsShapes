from itertools import groupby
import numpy as np

states = ['0', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'j']

forbidden_transition_attempts = ['be', 'bf', 'bg',
                                 'di',
                                 'eb', 'ei',
                                 'fb', 'fi',
                                 'gb',
                                 'id', 'ie', 'if']

allowed_transition_attempts = ['ab', 'ad',
                               'ba',
                               'de', 'df', 'da',
                               'ed', 'eg',
                               'fd', 'fg',
                               'gf', 'ge', 'gj',
                               'ij',
                               'jg', 'ji']
final_state = 'j'


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
        labels = [conf_space_labeled.space_labeled[index] for index in indices]
        labels = self.cut_of_after_final_state(labels)
        labels = self.interpolate_zeros(labels)
        labels = self.clean(labels)
        labels = self.add_missing_transitions(labels)
        return labels

    @staticmethod
    def clean(labels):
        not_allowed = [('bg', 'gb')]
        last_state = labels[0]
        labels_copy = labels.copy()
        for ii, next_state in enumerate(labels[1:], start=1):
            if (last_state, next_state) == ('bg', 'gb'):
                DEBUG = 1
            if (last_state, next_state) in not_allowed:
                labels_copy[ii] = last_state
            last_state = labels_copy[ii]
        return labels_copy

    @staticmethod
    def cut_of_after_final_state(labels):
        first_time_in_final_state = np.where(np.array(labels) == final_state)[0]
        if len(first_time_in_final_state) > 1:
            return labels[:first_time_in_final_state[0]+1]
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
        return len(set(state1) & set(state2)) > 0 or state1 == state2

    def add_missing_transitions(self, labels) -> list:
        """
        I want to correct states series, that are [.... 'g' 'b'...] to [... 'g' 'gb' 'b'...]
        """
        labels_copy = labels.copy()
        i = 1
        for ii, (state1, state2) in enumerate(zip(labels[:-1], labels[1:])):
            if self.valid_transition(state1, state2):
                pass

            elif len(state1) == len(state2) == 1:
                transition = ''.join(sorted(state1 + state2))
                if transition in allowed_transition_attempts:
                    labels_copy.insert(ii+i, transition)
                    i += 1
                else:
                    raise ValueError('What happened0?')
            elif len(state1) == len(state2) == 2:
                raise ValueError('What happened1?')

            elif ''.join(sorted(state1 + state2[0])) in allowed_transition_attempts:
                t2 = ''.join(sorted(state1 + state2[0]))
                t1 = state2[0]
                labels_copy.insert(ii + i, t1)
                labels_copy.insert(ii + i + 1, t2)
                i += 2
            elif ''.join(sorted(state1[0] + state2)) in allowed_transition_attempts:
                t1 = state1[0]
                t2 = ''.join(sorted(state1[0] + state2))
                labels_copy.insert(ii + i, t1)
                labels_copy.insert(ii + i + 1, t2)
                i += 2
            elif ''.join(sorted(state1[1] + state2)) in allowed_transition_attempts:
                t1 = state1[1]
                t2 = ''.join(sorted(state1[1] + state2))
                labels_copy.insert(ii + i, t1)
                labels_copy.insert(ii + i + 1, t2)
                i += 2
            else:
                raise ValueError('What happened2?')
        return labels_copy

    def state_at_time(self, time: float) -> str:
        return self.time_series[int(time/self.time_step)]

    @staticmethod
    def cut_at_end(time_series) -> list:
        """
        After state 'j' appears, cut off series
        :param time_series: series to be mashed
        :return: time_series with combined transitions
        """
        if 'j' not in time_series:
            return time_series
        first_appearance = np.where(np.array(time_series) == 'j')[0][0]
        return time_series[:first_appearance+1]

    @staticmethod
    def interpolate_zeros(labels: list) -> list:
        """
        Interpolate over all the states, that are not inside Configuration space (due to the computer representation of
        the maze not being exactly the same as the real maze)
        :return:
        """
        if labels[0] == '0':
            labels[0] = [l for l in labels if l != '0'][:1000][0]
        for i, l in enumerate(labels):
            if l == '0':
                labels[i] = labels[i - 1]
        return labels

    def calculate_state_series(self):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        labels = self.combine_transitions(labels)
        labels = self.cut_of_after_final_state(labels)
        return labels

    def time_stamped_series(self) -> list:
        groups = groupby(self.time_series)
        return [(label, sum(1 for _ in group) * self.time_step) for label, group in groups]
