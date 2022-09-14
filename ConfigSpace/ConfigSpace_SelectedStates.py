from ConfigSpace.ConfigSpace_Maze import *

# same_names = [['e', 'd'], ['eb', 'db'], ['ec', 'dc'], ['ef', 'df'], ['eg', 'dg'], ['ce', 'cd'], ['fe', 'fd'],
#               ['ge', 'gd'], ['be', 'bd']]

new_names = ['a1', 'a2', 'eg', 'be']

same_names = [
    # ['0', '0'],
    # ['a', 'a'],
    ['ab', 'a'],
    ['ac', 'a'],
    ['b', 'b'],
    ['ba', 'b'],
    ['bd', 'be'],
    # ['be', 'be'],
    ['bf', 'bf'],
    # ['c', 'c'],
    ['ca', 'c'],
    ['cd', 'c'],
    ['ce', 'c'],
    ['cg', 'c'],
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
    ['f', 'f'],
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

states = {'a', 'b', 'be', 'bf', 'c', 'e', 'eb', 'eg', 'f', 'g', 'h'}

connected = [['a1', 'a2', 'b', 'be'], ['b', 'bf', 'be'], ['a1', 'a2', 'c'], ['c', 'e'], ['eb', 'e'], ['e', 'f'],
             ['f', 'h'], ['h', 'g']]


class ConfigSpace_AdditionalStates(ConfigSpace_Labeled):
    def __init__(self, solver, size, shape, geometry):
        super().__init__(solver=solver, size=size, shape=shape, geometry=geometry)

    def reduce_states(self):
        for name_initial, name_final in same_names:
            self.space_labeled[name_initial == self.space_labeled] = name_final

        # split 'a'
        boundary = self.space_labeled.shape[2] // 2
        b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
        b_mask[..., boundary:] = True
        a_mask = np.isin(self.space_labeled, ['a', 'ab', 'ac'])

        self.space_labeled[np.logical_and(a_mask, b_mask)] = 'a1'
        self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'a2'

    def enlarge_transitions(self):
        to_enlarge = {'be': 30, 'bf': 30, 'eb': 45, 'eg': 45}
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

        else:
            raise ValueError('Cannot find ' + directory)

    @staticmethod
    def valid_state_transition(s1, s2):
        for c in connected:
            if s1 in c and s2 in c:
                print('yes')
        k = 1
        return True

if __name__ == '__main__':
    DEBUG = 1
    # solver, size, shape, geometry = ('ant', 'M', 'SPT',
    #                                  ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    solver, size, shape, geometry = ('human', 'Medium', 'SPT',
                                     ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))

    cs_labeled = ConfigSpace_AdditionalStates(solver, size, shape, geometry)
    cs_labeled.load_labeled_space()
    cs_labeled.visualize_transitions(reduction=2)

    DEBUG = 1
    k = 1
