from ConfigSpace.ConfigSpace_Maze import *
from os import path

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

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
connected = [['ab', 'ac'], ['ab', 'b', 'be', 'b1', 'b2'], ['ac', 'c'], ['c', 'e', 'cg'], ['eb', 'e'], ['e', 'f'],
             ['e', 'eg'], ['f', 'h'], ['h', 'g', 'i']]


class ConfigSpace_SelectedStates(ConfigSpace_Labeled):
    def __init__(self, solver, size, shape, geometry):
        # if solver == 'gillespie':
        #     size = 'M'
        #     geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')
        super().__init__(solver=solver, size=size, shape=shape, geometry=geometry)
        self.adapt_to_new_dimensions = False

    # def reduce_states(self):
    #     for name_initial, name_final in same_names:
    #         self.space_labeled[name_initial == self.space_labeled] = name_final

    # I should not need this again
    # def split_states(self):
    #     # split 'a'
    #     boundary = self.space_labeled.shape[2] / 4
    #     b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
    #     b_mask[..., int(boundary):int(boundary * 3)] = True
    #     a_mask = np.isin(self.space_labeled, ['a', 'ab', 'ac'])
    #
    #     self.space_labeled[np.logical_and(a_mask, b_mask)] = 'ac'
    #     self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'ab'
    #
    #     # split 'bf'
    #     a_mask = self.space_labeled == 'bf'
    #     boundary = self.space_labeled.shape[1] // 2 + 1
    #     b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
    #     b_mask[:, :int(boundary)] = True
    #
    #     self.space_labeled[np.logical_and(a_mask, b_mask)] = 'b1'
    #     self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'b2'

    # I should not need this again
    # def enlarge_transitions(self):
    #     # to_enlarge = {'be': 30, 'bf': 30, 'eb': 45, 'eg': 45, 'cg': 15}
    #     to_enlarge = {'bf': 30}
    #     for state, radius in to_enlarge.items():
    #         mask = self.dilate(self.space_labeled == state, radius)
    #         mask = np.logical_and(mask, self.space_labeled == state[0])
    #         self.space_labeled[mask] = state

    def visualize_transitions(self, reduction: int = 2, fig=None, colormap: str = 'Greys', space: np.ndarray = None,
                              only_states=None) \
            -> None:
        """

        :param fig: mylab figure reference
        :param reduction: What amount of reduction?
        :param space:
        :param colormap:
        :return:
        """
        if self.fig is None or not self.fig.running:
            self.fig = self.new_fig()
            # self.visualize_space(reduction=reduction)
        else:
            self.fig = fig

        if self.space_labeled is None:
            self.load_final_labeled_space()

        if only_states is None:
            ps_states_to_draw = [trans for trans in np.unique(self.space_labeled)]
            ps_states_to_draw.remove('0')
        else:
            ps_states_to_draw = only_states
        print('Draw transitions: ', str(ps_states_to_draw))

        for label, colormap in tqdm(zip(ps_states_to_draw, itertools.cycle(['Reds', 'Purples', 'Greens']))):
            space = np.array(self.space_labeled == label, dtype=bool)
            centroid = self.indices_to_coords(*np.array(np.where(space))[:, 0])
            self.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction, space=space)
            # mlab.text3d(*(a * b for a, b in zip(centroid, [1, 1, self.average_radius])), label,
            #             scale=self.scale_of_letters(reduction))
        # self.draw_ind((194, 149,  62))
        # self.draw_ind((194, 103, 554))
        # DEBUG = 1

    def load_final_labeled_space(self):
        if self.geometry == ('MazeDimensions_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'):
            geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')
            self.adapt_to_new_dimensions = True
            size = self.size
        elif self.geometry == ('MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx',
                               'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx') and self.solver == 'gillespie':
            geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')
            size = 'M'
        else:
            geometry = self.geometry
            size = self.size

        if self.size in ['Small Far', 'Small Near']:  # both have same dimensions
            filename = 'Small' + '_' + self.shape + '_' + geometry[0][:-5]

        else:
            filename = size + '_' + self.shape + '_' + geometry[0][:-5]

        directory = path.join(PhaseSpaceDirectory, self.shape, filename + '_labeled_final.pkl')
        if not path.exists(directory):
            raise ValueError('File does not exist: ', directory)
        print('Loading labeled from ', directory, '.')
        self.space_labeled = pickle.load(open(directory, 'rb'))
        print('finished loading')

    # I should never need this function again...
    def load_labeled_space(self, point_particle: bool = False) -> None:
        raise ValueError('use load_final_labeled_space')

    #     """
    #     Load Phase Space pickle.
    #     param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
    #     """
    #     directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius, small=True)
    #
    #     if path.exists(directory):
    #         print('Loading labeled from ', directory, '.')
    #         # self.space_labeled = pickle.load(open(path, 'rb'))
    #         self.space_labeled = pickle.load(open(directory, 'rb'))
    #         self.reduce_states()
    #         # self.enlarge_transitions()
    #         self.split_states()
    #
    #     else:
    #         raise ValueError('Cannot find ' + directory)

    @staticmethod
    def valid_state_transition(s1, s2) -> bool:
        for c in connected:
            if s1 in c and s2 in c:
                return True
        return False

    def save_final_labeled(self):
        if self.size in ['Small Far', 'Small Near']:  # both have same dimensions
            filename = 'Small' + '_' + self.shape + '_' + self.geometry[0][:-5]
        else:
            filename = self.size + '_' + self.shape + '_' + self.geometry[0][:-5]

        directory = path.join(PhaseSpaceDirectory, self.shape, filename + '_labeled_final.pkl')
        pickle.dump(self.space_labeled, open(directory, 'wb'))

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
                if state1 in ['f', 'e'] and state2 == 'i':
                    new_labels.append(state1)  # only for small SPT ants
                elif state1 in ['eg', 'dg', 'cg'] and state2 == 'g':
                    new_labels.append(state1)  # only for small SPT ants
                elif state1 == 'ba' and state2 in ['d', 'e']:
                    new_labels.append(state1)
                elif state1 in ['b2', 'b1', 'f'] and state2 in ['b2', 'b1', 'f']:
                    new_labels.append(state1)
                elif state1 in ['c', 'g', 'gc', 'cg'] and state2 in ['g', 'c', 'gc', 'cg']:
                    new_labels.append(state1)
                elif state1 in ['be', 'eb', 'gc', 'cg'] and state2 in ['eb', 'be', 'gc', 'cg']:
                    new_labels.append(state1)
                elif state1 in ['b', 'e', 'be', 'eb'] and state2 in ['b', 'e', 'be', 'eb']:
                    new_labels.append(state1)
                elif state1 in ['b1', 'b2'] and state2 in ['e']:
                    new_labels.append(state1)

                # elif len(state2) == 2 and state1 == state2[1]:
                #     new_labels.append(state2[1] + state2[0])
                else:
                    for t in cls.necessary_transitions(state1, state2, ii=ii):
                        new_labels.append(t)
                    new_labels.append(state2)
            else:
                new_labels.append(state2)
        return new_labels


def fix_cg():
    x_min = maze.slits[1] - maze.getLoadDim()[1] * (centerOfMass_shift + 1 / 2)
    # maze.set_configuration(position=(x_min, maze.arena_height / 2), angle=np.pi)
    # maze.draw()

    y_min = (-maze.exit_size / 2 + maze.arena_height / 2 + maze.getLoadDim()[2] / 2)
    # maze.set_configuration(position=(x_min, y_min), angle=np.pi)
    # maze.draw()

    xi, yi, thetai = ps.coords_to_indices(x_min, maze.arena_height / 2, np.pi)
    _, y_radius, _ = ps.coords_to_indices(x_min, y_min, np.pi)
    radius = yi - y_radius
    ps.space_labeled[ps.space_labeled == 'cg'] = 'c'

    mask_cg = create_circular_mask(ps.space_labeled.shape, center=(xi, yi, thetai), radius=2 * radius)
    mask_cg = np.logical_and(mask_cg, ps.space_labeled == 'c')
    ps.space_labeled[mask_cg] = 'cg'


if __name__ == '__main__':

    geometries = {
        ('ant', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')): ['XL', 'L', 'M', 'S'],
        ('human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')): ['Large', 'Medium', 'Small Far'],
        ('humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')): ['']
    }

    for (solver, geometry), sizes in list(geometries.items()):
        for size in sizes:
            maze = Maze(size=size, shape='SPT', solver=solver, geometry=geometry)
            ps = ConfigSpace_SelectedStates(solver, size, 'SPT', geometry)
            ps.load_final_labeled_space()
            # fix_cg()
            ps.visualize_space()
            # ps.visualize_transitions(only_states=['cg'], reduction=2)
            # ps.visualize_transitions()
            # ps.save_final_labeled()
            DEBUG = 1
