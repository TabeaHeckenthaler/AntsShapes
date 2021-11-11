import numpy as np
from Box2D import b2BodyDef, b2_staticBody, b2World, b2_dynamicBody, b2FixtureDef, b2CircleShape, b2Vec2
from Setup.MazeFunctions import BoxIt
from scipy.spatial import cKDTree
from pandas import read_excel
from Directories import home
from PhysicsEngine.drawables import Polygon, Point, Circle, colors
from copy import copy

ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie']  # also in Maze.py

periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}
ASSYMETRIC_H_SHIFT = 1.22 * 2
SPT_RATIO = 2.44 / 4.82  # ratio between shorter and longer side on the Special T
centerOfMass_shift = - 0.10880829015544041  # shift of the center of mass away from the center of the load.

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
              'SPT': [0, 1, 2, 3, 4, 5, 6], 'LASH': [0, 1, 2, 3, 4, 5, 6], 'RASH': [0, 1, 2, 3, 4, 5, 6],
              'circle': [0]}

ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'ps_simulation': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'human': {'Small Near': 1, 'Small Far': 1, 'S': 1, 'M': 1, 'Medium': 1, 'Large': 1, 'L': 1},
                 'humanhand': {'': 1}}

for solver in ant_dimensions:
    ResizeFactors[solver] = ResizeFactors['ant']


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
    def __init__(self, *args, size='XL', shape='SPT', solver='ant', free=False, position=None, angle=0,
                 point_particle=False, new2021=False):
        super().__init__(gravity=(0, 0), doSleep=True)

        if len(args) > 0 and type(args[0]).__name__ in ['Trajectory_human', 'Trajectory_ps_simulation',
                                                         'Trajectory_ant', 'Trajectory_gillespie', 'Trajectory']:

            self.shape = args[0].shape  # loadshape (maybe this will become name of the maze...)
            self.size = args[0].size  # size
            self.solver = args[0].solver
            if position is None:
                position = args[0].position[0]
            if position is None:
                position = args[0].angle[0]

        else:
            self.shape = shape  # loadshape (maybe this will become name of the maze...)
            self.size = size  # size
            self.solver = solver

        self.free = free
        self.new2021 = new2021
        self.statenames = StateNames[shape]
        self.getMazeDim()
        self.body = self.CreateMaze()
        self.get_zone()

        self.create_Load(position=position, angle=angle, point_particle=point_particle)

    def getMazeDim(self):
        if self.free:
            self.arena_height = 10
            self.arena_length = 10
            return

        else:
            dir = home + '\\Setup'

            if self.solver == 'ant' and self.new2021:
                df = read_excel(dir + '\\MazeDimensions_new2021_ant.xlsx', engine='openpyxl')
            elif self.solver in ['sim', 'gillespie']:
                df = read_excel(dir + '\\MazeDimensions_ant.xlsx', engine='openpyxl')
            else:
                df = read_excel(dir + '\\MazeDimensions_' + self.solver + '.xlsx', engine='openpyxl')

            if self.solver in ['ant', 'ps_simulation', 'sim', 'gillespie']:  # all measurements in cm
                d = df.loc[df['Name'] == self.size + '_' + self.shape]
                if hasattr(self, 'different_dimensions') and self.different_dimensions:
                    d = df.loc[df['Name'] == 'L_I1']  # these are special maze dimensions

                self.arena_length = d['arena_length'].values[0]
                self.arena_height = d['arena_height'].values[0]
                self.exit_size = d['exit_size'].values[0]
                self.wallthick = d['wallthick'].values[0]
                if type(d['slits'].values[0]) == str:
                    self.slits = [float(s) for s in d['slits'].values[0].split(', ')]
                else:
                    self.slits = [d['slits'].values[0]]

            elif self.solver == 'human' and not self.new2021:  # all measurements in meters
                # StartedScripts: measure the slits again...
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

    def CreateMaze(self):
        my_maze = self.CreateBody(b2BodyDef(position=(0, 0), angle=0, type=b2_staticBody, userData='my_maze'))

        if self.free:
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
        if self.free:
            self.zone = np.empty([0, 2])
            return
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

    # def possible_state_transitions(self, From, To):
    #     transitions = dict()
    #
    #     s = self.statenames
    #     if self.shape == 'H':
    #         transitions[s[0]] = [s[0], s[1], s[2]]
    #         transitions[s[1]] = [s[1], s[0], s[2], s[3]]
    #         transitions[s[2]] = [s[2], s[0], s[1], s[4]]
    #         transitions[s[3]] = [s[3], s[1], s[4], s[5]]
    #         transitions[s[4]] = [s[4], s[2], s[3], s[5]]
    #         transitions[s[5]] = [s[5], s[3], s[4]]
    #         return transitions[self.states[-1]].count(To) > 0
    #
    #     if self.shape == 'SPT':
    #         transitions[s[0]] = [s[0], s[1]]
    #         transitions[s[1]] = [s[1], s[0], s[2]]
    #         transitions[s[2]] = [s[2], s[1], s[3]]
    #         transitions[s[3]] = [s[3], s[2], s[4]]
    #         transitions[s[4]] = [s[4], s[3], s[5]]
    #         transitions[s[5]] = [s[5], s[4], s[6]]
    #         transitions[s[6]] = [s[6], s[5]]
    #         return transitions[self.states[From]].count(To) > 0

    def set_configuration(self, position, angle):
        self.bodies[-1].position.x, self.bodies[-1].position.y, self.bodies[-1].angle = position[0], position[1], angle

    def minimal_path_length(self):
        from DataFrame.dataFrame import myDataFrame
        from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
        p = myDataFrame.loc[myDataFrame['filename'] == filename_dstar(self.size, self.shape, 0, 0)][['path length [length unit]']]
        return p.values[0][0]

    def create_Load(self, position=None, angle=0, point_particle=False):
        if position is None:
            position = [0, 0]
        self.CreateBody(b2BodyDef(position=(float(position[0]), float(position[1])),
                                  angle=float(angle),
                                  type=b2_dynamicBody,
                                  fixedRotation=False,
                                  linearDamping=0,
                                  angularDamping=0,
                                  userData='my_load'),
                        restitution=0,
                        friction=0,
                        )

        self.addLoadFixtures(point_particle=point_particle)

    def addLoadFixtures(self, point_particle=False):
        if point_particle:
            return

        my_load = self.bodies[-1]
        if self.shape == 'circle':
            from trajectory_inheritance.gillespie import radius
            my_load.CreateFixture(b2FixtureDef(shape=b2CircleShape(pos=(0, 0), radius=radius)),
                                  density=1, friction=0, restitution=0,
                                  )

        if self.shape == 'H':
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2),
                (shape_width / 2, -shape_thickness / 2),
                (-shape_width / 2, -shape_thickness / 2),
                (-shape_width / 2, shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2),
                (shape_width / 2, shape_height / 2),
                (shape_width / 2 - shape_thickness, shape_height / 2),
                (shape_width / 2 - shape_thickness, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2),
                (-shape_width / 2, shape_height / 2),
                (-shape_width / 2 + shape_thickness, shape_height / 2),
                (-shape_width / 2 + shape_thickness, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'I':
            [shape_height, _, shape_thickness] = self.getLoadDim()
            my_load.CreatePolygonFixture(vertices=[
                (shape_height / 2, -shape_thickness / 2),
                (shape_height / 2, shape_thickness / 2),
                (-shape_height / 2, shape_thickness / 2),
                (-shape_height / 2, -shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'T':
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            resize_factor = ResizeFactors[self.solver][self.size]
            h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower part of the T.

            #  Top horizontal T part
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, shape_width / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_width / 2)],
                density=1, friction=0, restitution=0,
            )

            #  Bottom vertical T part
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_thickness / 2)], density=1, friction=0, restitution=0)

        if self.shape == 'SPT':  # This is the Special T
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim(short_edge=True)

            # h = SPT_centroid_shift * ResizeFactors[x.size]  # distance of the centroid away from the center of the
            # long middle
            h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
            # part of the T. (1.445 calculated)

            # This is the connecting middle piece
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2 - h, shape_thickness / 2),
                (shape_width / 2 - h, -shape_thickness / 2),
                (-shape_width / 2 - h, -shape_thickness / 2),
                (-shape_width / 2 - h, shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

            # This is the short side
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2 - h, -short_edge / 2),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2 - h, short_edge / 2),
                (shape_width / 2 - shape_thickness - h, short_edge / 2),
                (shape_width / 2 - shape_thickness - h, -short_edge / 2)],
                density=1, friction=0, restitution=0,
            )

            # This is the long side
            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2 - h, -shape_height / 2),
                (-shape_width / 2 - h, shape_height / 2),
                (-shape_width / 2 + shape_thickness - h, shape_height / 2),
                (-shape_width / 2 + shape_thickness - h, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'RASH':  # This is the ASymmetrical H
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            assymetric_h_shift = ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][self.size]
            # I multiply all these values with 2, because I got them in L, but want to state
            # them in XL.
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2,),
                (shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, shape_thickness / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2, shape_height / 2,),
                (shape_width / 2 - shape_thickness, shape_height / 2,),
                (shape_width / 2 - shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2,),
                (-shape_width / 2, shape_height / 2 - assymetric_h_shift,),
                (-shape_width / 2 + shape_thickness, shape_height / 2 - assymetric_h_shift,),
                (-shape_width / 2 + shape_thickness, -shape_height / 2,)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'LASH':  # This is the ASymmetrical H
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            assymetric_h_shift = ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][self.size]
            # I multiply all these values with 2, because I got them in L, but want to state
            # them in XL.
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2,),
                (shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, shape_thickness / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2,),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2, shape_height / 2 - assymetric_h_shift,),
                (shape_width / 2 - shape_thickness, shape_height / 2 - assymetric_h_shift,),
                (shape_width / 2 - shape_thickness, -shape_height / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
                (-shape_width / 2, shape_height / 2,),
                (-shape_width / 2 + shape_thickness, shape_height / 2,),
                (-shape_width / 2 + shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
                density=1, friction=0, restitution=0,
            )
        return my_load

    def getLoadDim(self, short_edge=False):
        if self.solver in ant_dimensions:
            resize_factor = ResizeFactors[self.solver][self.size]
            shape_sizes = {'H': [5.6, 7.2, 1.6],
                           'SPT': [4.85, 9.65, 0.85],
                           'LASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                           'RASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                           'I': [5.5, 1.75, 1.75],
                           'T': [5.4, 5.6, 1.6]
                           }
            if short_edge:
                shape_sizes['SPT'] = [4.85, 9.65, 0.85, 4.85 * SPT_RATIO]
            # dimensions = [shape_height, shape_width, shape_thickness, optional: long_edge/short_edge]
            dimensions = [i * resize_factor for i in shape_sizes[self.shape]]

            if (resize_factor == 1) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [8.14, 5.6, 1.2]]
            elif (resize_factor == 0.75) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [9, 6.2, 1.2]]
            return dimensions

        elif self.solver == 'human':
            # [shape_height, shape_width, shape_thickness, short_edge]
            if short_edge:
                SPT_Human_sizes = {'S': [0.805, 1.61, 0.125, 0.405],
                                   'M': [1.59, 3.18, 0.240, 0.795],
                                   'L': [3.2, 6.38, 0.51, 1.585]}
            else:
                SPT_Human_sizes = {'S': [0.805, 1.61, 0.125],
                                   'M': [1.59, 3.18, 0.240],
                                   'L': [3.2, 6.38, 0.51]}
            return SPT_Human_sizes[self.size[0]]

        elif self.solver == 'humanhand':
            # [shape_height, shape_width, shape_thickness, short_edge]
            SPT_Human_sizes = [6, 12.9, 0.9, 3]
            if not short_edge:
                SPT_Human_sizes = SPT_Human_sizes[:3]
            return SPT_Human_sizes

    def force_attachment_positions_in_trajectory(self, x):
        initial_pos, initial_angle = copy(self.bodies[-1].position), copy(self.bodies[-1].angle)
        force_attachment_positions_in_trajectory = []
        for i in range(len(x.frames)):
            self.set_configuration(x.position[i], x.angle[i])
            force_attachment_positions_in_trajectory.append(self.force_attachment_positions())
        self.set_configuration(initial_pos, initial_angle)
        return np.array(force_attachment_positions_in_trajectory)

    def force_attachment_positions(self):
        from trajectory_inheritance.humans import participant_number
        if self.solver == 'human' and self.size == 'Medium' and self.shape == 'SPT':
            # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim(short_edge=True)
            x29, x38, x47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4

            # (0, 0) is the middle of the shape
            positions = [[shape_width / 2, 0],
                         [x29, shape_thickness / 2],
                         [x38, shape_thickness / 2],
                         [x47, shape_thickness / 2],
                         [-shape_width / 2, shape_height / 4],
                         [-shape_width / 2, -shape_height / 4],
                         [x47, -shape_thickness / 2],
                         [x38, -shape_thickness / 2],
                         [x29, -shape_thickness / 2]]
            h = centerOfMass_shift * shape_width

        elif self.solver == 'human' and self.size == 'Large' and self.shape == 'SPT':
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim(short_edge=True)

            xMNOP = -shape_width / 2
            xLQ = xMNOP + shape_thickness / 2
            xAB = (-1) * xMNOP
            xCZ = (-1) * xLQ
            xKR = xMNOP + shape_thickness
            xJS, xIT, xHU, xGV, xFW, xEX, xDY = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

            yA_B = short_edge / 6
            yC_Z = short_edge / 2
            yDEFGHIJ_STUVWXY = shape_thickness / 2
            yK_R = shape_height / 10 * 3  # TODO: Tabea, you changed this
            yL_Q = shape_height / 2
            yM_P = shape_height / 10 * 3
            yN_O = shape_height / 10

            # indices in comment describe the index shown in Aviram's tracking movie
            positions = [[xAB, -yA_B],  # 1, A
                         [xAB, yA_B],  # 2, B
                         [xCZ, yC_Z],  # 3, C
                         [xDY, yDEFGHIJ_STUVWXY],  # 4, D
                         [xEX, yDEFGHIJ_STUVWXY],  # 5, E
                         [xFW, yDEFGHIJ_STUVWXY],  # 6, F
                         [xGV, yDEFGHIJ_STUVWXY],  # 7, G
                         [xHU, yDEFGHIJ_STUVWXY],  # 8, H
                         [xIT, yDEFGHIJ_STUVWXY],  # 9, I
                         [xJS, yDEFGHIJ_STUVWXY],  # 10, J
                         [xKR, yK_R],  # 11, K
                         [xLQ, yL_Q],  # 12, L
                         [xMNOP, yM_P],  # 13, M
                         [xMNOP, yN_O],  # 14, N
                         [xMNOP, -yN_O],  # 15, O
                         [xMNOP, -yM_P],  # 16, P
                         [xLQ, -yL_Q],  # 17, Q
                         [xKR, -yK_R],  # 18, R
                         [xJS, -yDEFGHIJ_STUVWXY],  # 19, S
                         [xIT, -yDEFGHIJ_STUVWXY],  # 20, T
                         [xHU, -yDEFGHIJ_STUVWXY],  # 21, U
                         [xGV, -yDEFGHIJ_STUVWXY],  # 22, V
                         [xFW, -yDEFGHIJ_STUVWXY],  # 23, W
                         [xEX, -yDEFGHIJ_STUVWXY],  # 24, X
                         [xDY, -yDEFGHIJ_STUVWXY],  # 25, Y
                         [xCZ, -yC_Z],  # 26, Z
                         ]
            h = centerOfMass_shift * shape_width

        else:
            positions = [[0, 0] for i in range(participant_number[self.size])]
            h = 0

        # centerOfMass_shift the shape...
        positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame
        return np.array(
            [np.array(self.bodies[-1].GetWorldPoint(b2Vec2(r))) for r in positions])  # r vectors in the lab frame

    def draw(self, display):
        for body in self.bodies:
            for fixture in body.fixtures:
                if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
                    vertices = [(body.transform * v) for v in fixture.shape.vertices]
                    Polygon(vertices, color=colors[body.userData]).draw(display)

                elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
                    position = body.position + fixture.shape.pos
                    Circle(position, fixture.shape.radius, color=colors[body.userData]).draw(display)

            if body.userData == 'my_load':
                Point(np.array(body.position)).draw(display)

    def average_radius(self):
        r = ResizeFactors[self.solver][self.size]
        radii = {'H': 2.9939 * r,
                 'I': 2.3292 * r,
                 'T': 2.9547 * r,
                 'SPT': 0.76791 * self.getLoadDim()[1],
                 'RASH': 2 * 1.6671 * r,
                 'LASH': 2 * 1.6671 * r}
        return radii[self.shape]

    def circumference(self):
        shape_height, shape_width, shape_thickness = self.getLoadDim()

        if self.shape.endswith('ASH'):
            print('I dont know circumference of ASH!!!')
            breakpoint()
        cir = {'H': 4 * shape_height - 2 * shape_thickness + 2 * shape_width,
               'I': 2 * shape_height + 2 * shape_width,
               'T': 2 * shape_height + 2 * shape_width,
               'SPT': 2 * shape_height * SPT_RATIO +
                      2 * shape_height -
                      2 * shape_thickness +
                      2 * shape_width,
               'RASH': 2 * shape_width + 4 * shape_height - 4 * ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][
                   self.size]
                       - 2 * shape_thickness,
               'LASH': 2 * shape_width + 4 * shape_height - 4 * ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][
                   self.size]
                       - 2 * shape_thickness
               }
        return cir[self.shape]
