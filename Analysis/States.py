from itertools import groupby


class States:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """
    def __init__(self, conf_space_labeled, x, step: int = 1):
        """
        :param step: after how many frames I add a label to my label list
        :param x: trajectory
        :return: list of strings with labels
        """
        self.time_step = step/x.fps
        indices = [conf_space_labeled.coords_to_indexes(*coords) for coords in x.iterate_coords(step=step)]
        self.time_series = [conf_space_labeled.space_labeled[index][0] for index in indices]
        self.interpolate_zeros()
        self.permitted_transitions = {'a': ['a', 'b', 'd'], 'b': ['b', 'a'], 'd': ['d', 'a', 'f', 'e'],
                                      'e': ['e', 'd', 'g'], 'f': ['f', 'd', 'g'], 'g': ['g', 'f', 'j'], 'i': ['i', 'j'],
                                      'j': ['j', 'i', 'g']}
        self.state_series = self.calculate_state_series()
        if len(self.transitions_forbidden()) > 0:
            print('Forbidden:', self.transitions_forbidden(), 'in', x.filename)
            print('You might want to decrease your step size, because you might be skipping state transitions.')

    def interpolate_zeros(self) -> None:
        """
        Interpolate over all the states, that are not inside Configuration space (due to the computer representation of
        the maze not being exactly the same as the real maze)
        :return:
        """
        if self.time_series[0] == '0':
            self.time_series[0] = [l for l in self.time_series if l != '0'][:1000][0]
        for i, l in enumerate(self.time_series):
            if l == '0':
                self.time_series[i] = self.time_series[i - 1]

    def transitions_forbidden(self) -> list:
        """
        Check whether the permitted transitions are all allowed
        :return: boolean, whether all transitions are allowed
        """
        return [str(l0) + ' to ' + str(l1) for l0, l1 in zip(self.time_series, self.time_series[1:])
                if l1 not in self.permitted_transitions[l0]]

    def calculate_state_series(self):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        return labels
