import numpy as np
from Box2D import b2BodyDef, b2_staticBody, b2World
from Setup.MazeFunctions import BoxIt
from scipy.spatial import cKDTree
from pandas import read_excel

size_per_shape = {'ant': {'H': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'I': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'T': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'SPT': ['S', 'M', 'L', 'XL'],
                          'RASH': ['S', 'M', 'L', 'XL'],
                          'LASH': ['S', 'M', 'L', 'XL'],
                          },
                  'human': {'SPT': ['S', 'M', 'L']},
                  'humanhand': {'SPT': ['']}
                  }

StateNames = {'H': [0, 1, 2, 3, 4, 5], 'I': [0, 1, 2, 3, 4, 5], 'T': [0, 1, 2, 3, 4, 5],
              'SPT': [0, 1, 2, 3, 4, 5, 6], 'LASH': [0, 1, 2, 3, 4, 5, 6], 'RASH': [0, 1, 2, 3, 4, 5, 6]}

ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'dstar': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'human': {'Small Near': 1, 'Small Far': 1, 'S': 1, 'M': 1, 'Medium': 1, 'Large': 1, 'L': 1},
                 'humanhand': {'': 1}}


# there are a few I mazes, which have a different exit size,

# x, y, theta
def start(size, shape, solver):
    maze = Maze(size=size, shape=shape, solver=solver)

    if shape == 'SPT':
        # return [(maze.slits[0] - maze.slits[-1]) / 2 + maze.slits[-1] - 0.5, maze.arena_height / 2, 0]
        return [maze.slits[0] * 0.5, maze.arena_height / 2, 0]
    elif shape in ['H', 'I', 'T', 'RASH', 'LASH']:
        return [maze.slits[0] - 5, maze.arena_height / 2, np.pi - 0.1]


def end(size, shape, solver):
    maze = Maze(size=size, shape=shape, solver=solver)
    return [maze.slits[-1] + 5, maze.arena_height / 2, 0]


class Maze(b2World):
    def __init__(self, *args, size='XL', shape='SPT', solver='ant', free=False):
        super().__init__(gravity=(0, 0), doSleep=True)
        self.shape = shape  # loadshape (maybe this will become name of the maze...)
        self.size = size  # size
        self.solver = solver
        self.statenames = StateNames[shape]
        self.getMazeDim(*args)
        self.body = self.CreateMaze(free)
        self.get_zone()

    def getMazeDim(self, *args):
        df = read_excel('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Setup\\MazeDimensions_' + self.solver + '.xlsx',
                        engine='openpyxl')
        if self.solver in ['ant', 'dstar']:  # all measurements in cm
            d = df.loc[df['Name'] == self.size + '_' + self.shape]

            if 'L_I1' in args:
                d = df.loc[df['Name'] == 'L_I1']

            self.arena_length = d['arena_length'].values[0]
            self.arena_height = d['arena_height'].values[0]
            self.exit_size = d['exit_size'].values[0]
            self.wallthick = d['wallthick'].values[0]
            if type(d['slits'].values[0]) == str:
                self.slits = [float(s) for s in d['slits'].values[0].split(', ')]
            else:
                self.slits = [d['slits'].values[0]]

        elif self.solver == 'human':  # all measurements in meters
            # TODO: measure the slits again...
            # these coordinate values are given inspired from the drawing in \\phys-guru-cs\ants\Tabea\Human
            # Experiments\ExperimentalSetup
            d = df.loc[df['Name'] == self.size]
            A = [float(s) for s in d['A'].values[0].split(',')]
            # B = [float(s) for s in d['B'].values[0].split(',')]
            C = [float(s) for s in d['C'].values[0].split(',')]
            D = [float(s) for s in d['D'].values[0].split(',')]
            E = [float(s) for s in d['E'].values[0].split(',')]

            self.arena_length, self.exit_size = A[0], D[1] - C[1]
            self.wallthick = 0.1
            self.arena_height = 2 * C[1] + self.exit_size
            self.slits = [(E[0] + self.wallthick / 2),
                          (C[0] + self.wallthick / 2)]  # These are the x positions at which the slits are positions

        elif self.solver == 'humanhand':  # only SPT
            d = df.loc[df['Name'] == self.solver]
            self.arena_length = d['arena_length'].values[0]
            self.arena_height = d['arena_height'].values[0]
            self.exit_size = d['exit_size'].values[0]
            self.wallthick = d['wallthick'].values[0]
            self.slits = [float(s) for s in d['slits'].values[0].split(', ')]

        self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)

    def CreateMaze(self, free):
        my_maze = self.CreateBody(b2BodyDef(position=(0, 0), angle=0, type=b2_staticBody, userData='my_maze'))
        if free:
            my_maze.CreateLoopFixture(
                vertices=[(0, 0), (0, self.arena_height * 3), (self.arena_length * 3, self.arena_height * 3),
                          (self.arena_length * 3, 0)])
        else:
            my_maze.CreateLoopFixture(
                vertices=[(0, 0),
                          (0, self.arena_height),
                          (self.arena_length, self.arena_height),
                          (self.arena_length, 0),
                          ])
            self.CreateSlitObject(my_maze)
        return my_maze

    def CreateSlitObject(self, my_maze):
        # # The x and y position describe the point, where the middle (in x direction) of the top edge (y direction)
        # of the lower wall of the slit is...
        """ We need a special case for L_SPT because in the manufacturing the slits were not vertically glued. """

        if self.shape == 'LongT':
            pass
            # self.slitpoints[i]
        if self.shape == 'SPT':
            if self.size == 'L' and self.solver == 'ant':
                slitLength = 4.1
                # this is the left (inside), bottom Slit
                self.slitpoints[0] = np.array([[self.slits[0], 0],
                                               [self.slits[0], slitLength],
                                               [self.slits[0] + self.wallthick, slitLength],
                                               [self.slits[0] + self.wallthick, 0]]
                                              )
                # this is the left (inside), upper Slit
                self.slitpoints[1] = np.array([[self.slits[0] - 0.05, slitLength + self.exit_size],
                                               [self.slits[0] + 0.1, self.arena_height],
                                               [self.slits[0] + self.wallthick + 0.1, self.arena_height],
                                               [self.slits[0] + self.wallthick - 0.05, slitLength + self.exit_size]]
                                              )

                # this is the right (outside), lower Slit
                self.slitpoints[2] = np.array([[self.slits[1], 0],
                                               [self.slits[1] + 0.1, slitLength],
                                               [self.slits[1] + self.wallthick + 0.1, slitLength],
                                               [self.slits[1] + self.wallthick, 0]]
                                              )
                # this is the right (outside), upper Slit
                self.slitpoints[3] = np.array([[self.slits[1] + 0.2, slitLength + self.exit_size],
                                               [self.slits[1] + 0.2, self.arena_height],
                                               [self.slits[1] + self.wallthick + 0.2, self.arena_height],
                                               [self.slits[1] + self.wallthick + 0.2, slitLength + self.exit_size]]
                                              )

            # elif size == 'M' or size == 'XL'
            else:
                slitLength = (self.arena_height - self.exit_size) / 2
                # this is the left (inside), bottom Slit
                self.slitpoints[0] = np.array([[self.slits[0], 0],
                                               [self.slits[0], slitLength],
                                               [self.slits[0] + self.wallthick, slitLength],
                                               [self.slits[0] + self.wallthick, 0]]
                                              )
                # this is the left (inside), upper Slit
                self.slitpoints[1] = np.array([[self.slits[0], slitLength + self.exit_size],
                                               [self.slits[0], self.arena_height],
                                               [self.slits[0] + self.wallthick, self.arena_height],
                                               [self.slits[0] + self.wallthick, slitLength + self.exit_size]]
                                              )

                # this is the right (outside), lower Slit
                self.slitpoints[2] = np.array([[self.slits[1], 0],
                                               [self.slits[1], slitLength],
                                               [self.slits[1] + self.wallthick, slitLength],
                                               [self.slits[1] + self.wallthick, 0]]
                                              )
                # this is the right (outside), upper Slit
                self.slitpoints[3] = np.array([[self.slits[1], slitLength + self.exit_size],
                                               [self.slits[1], self.arena_height],
                                               [self.slits[1] + self.wallthick, self.arena_height],
                                               [self.slits[1] + self.wallthick, slitLength + self.exit_size]]
                                              )

            # slit_up
            my_maze.CreatePolygonFixture(vertices=self.slitpoints[0].tolist())
            my_maze.CreatePolygonFixture(vertices=self.slitpoints[2].tolist())

            # slit_down
            my_maze.CreatePolygonFixture(vertices=self.slitpoints[1].tolist())
            my_maze.CreatePolygonFixture(vertices=self.slitpoints[3].tolist())

        # this is for all the 'normal SPT Mazes', that have no manufacturing mistakes            
        else:
            self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)
            for i, slit in enumerate(self.slits):
                # this is the lower Slit
                self.slitpoints[2 * i] = np.array([[slit, 0],
                                                   [slit, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, 0]]
                                                  )

                my_maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i].tolist())

                # this is the upper Slit
                self.slitpoints[2 * i + 1] = np.array([[slit, (self.arena_height + self.exit_size) / 2],
                                                       [slit, self.arena_height],
                                                       [slit + self.wallthick, self.arena_height],
                                                       [slit + self.wallthick,
                                                        (self.arena_height + self.exit_size) / 2]]
                                                      )

                my_maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i + 1].tolist())

        # I dont want to have the vertical line at the first exit
        self.slitTree = BoxIt(np.array([[0, 0],
                                        [0, self.arena_height],
                                        [self.slits[-1], self.arena_height],
                                        [self.slits[-1], 0]]),
                              0.1, without='right')

        for slit_points in self.slitpoints:
            self.slitTree = np.vstack((self.slitTree, BoxIt(slit_points, 0.01)))
        self.slitTree = cKDTree(self.slitTree)

    def get_zone(self):
        if self.shape == 'SPT':
            self.zone = np.array([[0, 0],
                                  [0, self.arena_height],
                                  [self.slits[0], self.arena_height],
                                  [self.slits[0], 0]])
        else:
            RF = ResizeFactors[self.solver][self.size]
            self.zone = np.array(
                [[self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 - self.arena_height * RF / 2],
                 [self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 + self.arena_height * RF / 2],
                 [self.slits[0], self.arena_height / 2 + self.arena_height * RF / 2],
                 [self.slits[0], self.arena_height / 2 - self.arena_height * RF / 2]])
        return

    def possible_state_transitions(self, From, To):
        transitions = dict()

        s = self.statenames
        if self.shape == 'H':
            transitions[s[0]] = [s[0], s[1], s[2]]
            transitions[s[1]] = [s[1], s[0], s[2], s[3]]
            transitions[s[2]] = [s[2], s[0], s[1], s[4]]
            transitions[s[3]] = [s[3], s[1], s[4], s[5]]
            transitions[s[4]] = [s[4], s[2], s[3], s[5]]
            transitions[s[5]] = [s[5], s[3], s[4]]
            return transitions[self.states[-1]].count(To) > 0

        if self.shape == 'SPT':
            transitions[s[0]] = [s[0], s[1]]
            transitions[s[1]] = [s[1], s[0], s[2]]
            transitions[s[2]] = [s[2], s[1], s[3]]
            transitions[s[3]] = [s[3], s[2], s[4]]
            transitions[s[4]] = [s[4], s[3], s[5]]
            transitions[s[5]] = [s[5], s[4], s[6]]
            transitions[s[6]] = [s[6], s[5]]
            return transitions[self.states[From]].count(To) > 0

    def minimal_path_length(self):
        from DataFrame.create_dataframe import df
        from Classes_Experiment.mr_dstar import filename_dstar
        p = df.loc[df['filename'] == filename_dstar(self.size, self.shape, 0, 0)][['path length [length unit]']]
        return p.values[0][0]


def maze_corners(maze):
    corners = [[0, 0],
               [0, maze.arena_height],
               [maze.slits[-1] + 20, maze.arena_height],
               [maze.slits[-1] + 20, 0],
               ]
    return corners + list(np.resize(maze.slitpoints, (16, 2)))
