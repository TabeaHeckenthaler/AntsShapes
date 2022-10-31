from ConfigSpace.ConfigSpace_Maze import *

# same_names = [['e', 'd'], ['eb', 'db'], ['ec', 'dc'], ['ef', 'df'], ['eg', 'dg'], ['ce', 'cd'], ['fe', 'fd'],
#               ['ge', 'gd'], ['be', 'bd']]

new_names = ['ab', 'ac', 'eg', 'be', 'cg', 'b1', 'b2']

perfect_states = ['ab', 'ac', 'c', 'e', 'f', 'h', 'i']

same_names = [
    # ['0', '0'],
    # ['a', 'a'],
    ['ab', 'a'],
    ['ac', 'a'],
    # ['b', 'b'],
    ['ba', 'b'],
    ['bd', 'be'],
    # ['be', 'be'],
    # ['bf', 'bf'],
    # ['c', 'c'],
    ['ca', 'c'],
    ['cd', 'c'],
    ['ce', 'c'],
    # ['cg', 'c'],
    ['d', 'e'],
    ['db', 'eb'],
    ['dc', 'e'],
    ['df', 'e'],
    ['dg', 'eg'],
    # ['e', 'e'],
    ['eb', 'eb'],
    ['ec', 'e'],
    ['ef', 'e'],
    # ['eg', 'eg'],
    # ['f', 'f'],
    ['fb', 'f'],
    ['fd', 'f'],
    ['fe', 'f'],
    ['fh', 'f'],
    # ['g', 'g'],
    ['gc', 'g'],
    ['gd', 'g'],
    ['ge', 'g'],
    ['gh', 'g'],
    # ['h', 'h'],
    ['hf', 'h'],
    ['hg', 'h']]

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}

connected = [['ab', 'ac'], ['ab', 'b', 'be', 'b1', 'b2'], ['ac', 'c'], ['c', 'e', 'cg'], ['eb', 'e'], ['e', 'f'],
             ['e', 'eg'], ['f', 'h'], ['h', 'g', 'i']]


class ConfigSpace_AdditionalStates(ConfigSpace_Labeled):
    def __init__(self, solver, size, shape, geometry):
        super().__init__(solver=solver, size=size, shape=shape, geometry=geometry)

    def reduce_states(self):
        for name_initial, name_final in same_names:
            self.space_labeled[name_initial == self.space_labeled] = name_final

    def split_states(self):
        # split 'a'
        boundary = self.space_labeled.shape[2] / 4
        b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
        b_mask[..., int(boundary):int(boundary*3)] = True
        a_mask = np.isin(self.space_labeled, ['a', 'ab', 'ac'])

        self.space_labeled[np.logical_and(a_mask, b_mask)] = 'ac'
        self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'ab'

        # split 'bf'
        a_mask = self.space_labeled == 'bf'
        boundary = self.space_labeled.shape[1] // 2 + 1
        b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
        b_mask[:, :int(boundary)] = True

        self.space_labeled[np.logical_and(a_mask, b_mask)] = 'b1'
        self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'b2'

    def enlarge_transitions(self):
        to_enlarge = {'be': 30, 'bf': 30, 'eb': 45, 'eg': 45, 'cg': 15}
        for state, radius in to_enlarge.items():
            mask = self.dilate(self.space_labeled == state, radius)
            mask = np.logical_and(mask, self.space_labeled == state[0])
            self.space_labeled[mask] = state

    def visualize_transitions(self, reduction: int = 1, fig=None, colormap: str = 'Greys', space: np.ndarray = None) \
            -> None:
        """

        :param fig: mylab figure reference
        :param reduction: What amount of reduction?
        :param space:
        :param colormap:
        :return:
        """
        if self.fig is None or not self.fig.running:
            self.visualize_space(reduction=reduction)

        else:
            self.fig = fig

        if self.space_labeled is None:
            self.load_labeled_space()

        print('Draw transitions')
        transitions = [trans for trans in np.unique(self.space_labeled)]
        transitions.remove('0')
        for label, colormap in tqdm(zip(transitions, itertools.cycle(['Reds', 'Purples', 'Greens']))):
            space = np.array(self.space_labeled == label, dtype=bool)
            centroid = self.indices_to_coords(*np.array(np.where(space))[:, 0])
            self.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction, space=space)
            mlab.text3d(*(a * b for a, b in zip(centroid, [1, 1, self.average_radius])), label,
                        scale=self.scale_of_letters(reduction))

    def load_labeled_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculafted.
        """
        directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius, small=True)

        if os.path.exists(directory):
            print('Loading labeled from ', directory, '.')
            # self.space_labeled = pickle.load(open(path, 'rb'))
            self.space_labeled = pickle.load(open(directory, 'rb'))
            self.reduce_states()
            self.enlarge_transitions()
            self.split_states()

        else:
            raise ValueError('Cannot find ' + directory)

    @staticmethod
    def valid_state_transition(s1, s2) -> bool:
        for c in connected:
            if s1 in c and s2 in c:
                return True
        return False

    @classmethod
    def add_missing_transitions(cls, labels) -> list:
        """
        I want to correct states series, that are [.... 'g' 'b'...] to [... 'g' 'gb' 'b'...]
        """
        new_labels = [labels[0]]

        for ii, state2 in enumerate(labels[1:]):
            # if state1 in ['cg', 'ac'] and state2 in ['cg', 'ac'] and state1 != state2:
            #     DEBUG = 1
            state1 = new_labels[-1]
            if not cls.valid_state_transition(state1, state2):
                if state1 in ['ac'] and 'e' in state2:
                    new_labels.append('c')  # only for small SPT ants
                elif state1 != 'be':
                    DEBUG = 1
            else:
                new_labels.append(state2)
        return new_labels


if __name__ == '__main__':
    DEBUG = 1
    solver, size, shape, geometry = ('ant', 'S', 'SPT',
                                     ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    # solver, size, shape, geometry = ('human', 'Medium', 'SPT',
    #                                  ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

    cs_labeled = ConfigSpace_AdditionalStates(solver, size, shape, geometry)
    cs_labeled.load_labeled_space()
    cs_labeled.visualize_transitions(reduction=2)

    DEBUG = 1
    k = 1
