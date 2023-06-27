import numpy as np
from Box2D import b2BodyDef, b2_staticBody, b2World, b2_dynamicBody, b2FixtureDef, b2CircleShape, b2Vec2
from Setup.MazeFunctions import BoxIt
from scipy.spatial import cKDTree
from pandas import read_excel
from Directories import maze_dimension_directory, home
from PhysicsEngine.drawables import Polygon, Point, Circle, colors
from copy import copy
from os import path
from trajectory_inheritance.exp_types import is_exp_valid, centerOfMass_shift
import json
from typing import Union

ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie', 'pheidole']  # also in Maze.py

# TODO: x = get(myDataFrame.loc[429].filename).play() displays a maze, that does not make any sense!

ASSYMETRIC_H_SHIFT = 1.22 * 2
# SPT_RATIO = 2.44 / 4.82  # ratio between shorter and longer side on the Special T
# My PS are still for the original value below!!
# centerOfMass_shift = - 0.10880829015544041  # shift of the center of mass away from the center of the load.

# size_per_shape = {'ant': {'H': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
#                           'I': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
#                           'T': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
#                           'SPT': ['S', 'M', 'L', 'XL'],
#                           'RASH': ['S', 'M', 'L', 'XL'],
#                           'LASH': ['S', 'M', 'L', 'XL'],
#                           },
#                   'human': {'SPT': ['S', 'M', 'L']},
#                   'humanhand': {'SPT': ['']}
#                   }

with open(path.join(home, 'Setup', 'ResizeFactors.json'), "r") as read_content:
    ResizeFactors = json.load(read_content)

# ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
#                  'pheidole': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
#                  'human': {'Small Near': 1, 'Small Far': 1, 'Medium': 1, 'Large': 1},
#                  'humanhand': {'': 1}}
# ResizeFactors['ps_simulation'] = dict(ResizeFactors['ant'], **ResizeFactors['human'], **ResizeFactors['humanhand'])
# ResizeFactors['gillespie'] = dict(ResizeFactors['ant'], **ResizeFactors['human'], **ResizeFactors['humanhand'])


def start(x, initial_cond: str):
    if initial_cond not in ['back', 'front']:
        raise ValueError('You initial_cond is not valid.')
    maze = Maze(x)
    if x.shape == 'SPT':
        if initial_cond == 'back':
            # return [(maze.slits[0] - maze.slits[-1]) / 2 + maze.slits[-1] - 0.5, maze.arena_height / 2, 0]
            return [maze.slits[0] * 0.5, maze.arena_height / 2, 0]
        elif initial_cond == 'front':
            return [maze.slits[0] + (maze.slits[1] - maze.slits[0]) * 0.4, maze.arena_height / 2, 0]
    elif x.shape in ['H', 'I', 'T', 'RASH', 'LASH']:
        return [maze.slits[0] - 5, maze.arena_height / 2, np.pi - 0.1]


def end(x):
    """
    :param x: Trajectory object
    :return: list with coordinates of end of SPT
    """
    maze = Maze(x)
    return [maze.slits[-1] * 1.26, maze.arena_height / 2, 0]


class Maze_parent(b2World):
    def __init__(self, position=None, angle=0, point_particle=False, bb: bool = False, free=False):
        super().__init__(gravity=(0, 0), doSleep=True)

        if not hasattr(self, 'size'):
            self.size = 'XL'
        if not hasattr(self, 'shape'):
            self.shape = 'SPT'
        if not hasattr(self, 'solver'):
            self.solver = 'ant'
        if not hasattr(self, 'arena_height'):
            self.arena_height = 10
        if not hasattr(self, 'arena_length'):
            self.arena_length = 'XL'
        if not hasattr(self, 'excel_file_load'):
            if self.shape == 'SPT':
                self.excel_file_load = 'LoadDimensions_new2021_SPT_ant.xlsx'
            else:
                self.excel_file_load = 'LoadDimensions_ant.xlsx'

        self.maze = self.create_Maze()
        self.free = free
        self.create_Load(position=position, angle=angle, point_particle=point_particle, bb=bb)

    def create_Maze(self):
        pass

    def set_configuration(self, position, angle):
        self.bodies[-1].position.x, self.bodies[-1].position.y, self.bodies[-1].angle = position[0], position[1], angle

    def create_Load(self, position=None, angle=0, point_particle=False, bb: bool = False):

        if position is None:
            position = [0, 0]
        self.CreateBody(b2BodyDef(position=(float(position[0]), float(position[1])),
                                  angle=float(angle),
                                  type=b2_dynamicBody,
                                  fixedRotation=False,
                                  linearDamping=0,
                                  angularDamping=0,
                                  userData='load'),
                        restitution=0,
                        friction=0,
                        )

        self.addLoadFixtures(point_particle=point_particle, bb=bb)

    def addLoadFixtures(self, point_particle=False, bb: bool = False):
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
            h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower force_vector of
            # the T.

            #  Top horizontal T force_vector
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, shape_width / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_width / 2)],
                density=1, friction=0, restitution=0,
            )

            #  Bottom vertical T force_vector
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_thickness / 2)], density=1, friction=0, restitution=0)

        if self.shape == 'SPT':  # This is the Special T
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim()
            h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
            if bb:
                my_load.CreatePolygonFixture(vertices=[
                    (shape_width / 2 - h, shape_height / 2),
                    (shape_width / 2 - h, -shape_height / 2),
                    (-shape_width / 2 - h, -shape_height / 2),
                    (-shape_width / 2 - h, shape_height / 2)],
                    density=1, friction=0, restitution=0,
                )
                return

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

    def getLoadDim(self):
        df = read_excel(path.join(maze_dimension_directory, self.excel_file_load), engine='openpyxl')

        if self.shape != 'SPT' and self.solver in ant_dimensions:
            d = df.loc[df['Name'] == self.shape]
            shape_sizes = [d['height'].values[0], d['width'].values[0], d['thickness'].values[0]]
            resize_factor = ResizeFactors[self.solver][self.size]
            dimensions = [i * resize_factor for i in shape_sizes]

            if (resize_factor == 1) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [8.14, 5.6, 1.2]]
            elif (resize_factor == 0.75) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [9, 6.2, 1.2]]
            return dimensions

        # if self.excel_file_load in ['LoadDimensions_ant.xlsx']:
        #     d = df.loc[df['Name'] == self.size + '_' + self.shape]

        elif self.excel_file_load in ['LoadDimensions_ant_L_I_425.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx',
                                      'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx']:
            d = df.loc[df['Name'] == self.size + '_' + self.shape]

        elif self.excel_file_load in ['LoadDimensions_human.xlsx']:
            d = df.loc[df['Name'] == self.size[0]]

        elif self.excel_file_load in ['LoadDimensions_humanhand.xlsx']:
            d = df.loc[0]
            return [d['long edge'], d['length'], d['width'], d['short edge']]

        else:
            raise ValueError('Gave dimensions for ' + self.excel_file_load + ' but not matching ' +
                             ' '.join([self.solver, self.shape, self.size]))

        dimensions = [d['long edge'].values[0], d['length'].values[0], d['width'].values[0],
                      d['short edge'].values[0]]
        return dimensions

    def force_attachment_positions_in_trajectory(self, x, reference_frame='maze'):
        """
        force attachment in world coordinates
        """
        initial_pos, initial_angle = copy(self.bodies[-1].position), copy(self.bodies[-1].angle)
        if reference_frame == 'maze':
            force_attachment_positions_in_trajectory = []
            for i in range(len(x.frames)):
                self.set_configuration(x.position[i], x.angle[i])
                force_attachment_positions_in_trajectory.append(self.force_attachment_positions())
            self.set_configuration(initial_pos, initial_angle)
            return np.array(force_attachment_positions_in_trajectory)
        elif reference_frame == 'load':
            self.set_configuration([0, 0], 0)
            force_attachment = np.stack([self.force_attachment_positions() for _ in range(len(x.frames))])
            self.set_configuration(initial_pos, initial_angle)
            return np.array(force_attachment)
        else:
            raise ValueError('Unknown reference frame!')

    def force_attachment_positions(self):
        from trajectory_inheritance.humans import participant_number
        if self.solver == 'human' and self.size == 'Medium' and self.shape == 'SPT':
            # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
            [shape_height, shape_width, shape_thickness, _] = self.getLoadDim()
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
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim()

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

            # indices_to_coords in comment describe the index shown in Aviram's tracking movie
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

    def draw(self, display=None, points=[], color=None):
        if display is None:
            from PhysicsEngine.Display import Display
            d = Display('', 1, self)
        else:
            d = display
        for body in self.bodies:
            for fixture in body.fixtures:
                if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
                    vertices = [(body.transform * v) for v in fixture.shape.vertices]
                    Polygon(vertices, color=colors[body.userData]).draw(d)

                elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
                    position = body.position + fixture.shape.pos
                    Circle(position, fixture.shape.radius, color=colors[body.userData]).draw(d)

            if body.userData == 'load':
                Point(np.array(body.position), color=(251, 0, 0)).draw(d)

        for point in points:

            Point(np.array(point), color=color).draw(d)

        if display is None:
            d.display()

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
        if self.shape == 'SPT':
            shape_height, shape_width, shape_thickness, shape_height_short_edge = self.getLoadDim()
        else:
            shape_height, shape_width, shape_thickness = self.getLoadDim()
            shape_height_short_edge = np.NaN

        shift = ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][self.size]

        cir = {'H': 4 * shape_height - 2 * shape_thickness + 2 * shape_width,
               'I': 2 * shape_height + 2 * shape_width,
               'T': 2 * shape_height + 2 * shape_width,
               'SPT': 2 * shape_height_short_edge + 2 * shape_height - 2 * shape_thickness + 2 * shape_width,
               'RASH': 2 * shape_width + 4 * shape_height - 4 * shift - 2 * shape_thickness,
               'LASH': 2 * shape_width + 4 * shape_height - 4 * shift - 2 * shape_thickness
               }

        if self.shape.endswith('ASH'):
            raise ValueError('I do not know circumference of ASH!!!')

        return cir[self.shape]


class Maze(Maze_parent):
    def __init__(self, *args, size='XL', shape='SPT', solver='ant', position=None, angle=0, point_particle=False,
                 geometry: tuple = None, i=0, bb: bool = False):
        self.arena_length = float()
        self.arena_height = float()
        self.exit_size = float()
        self.wallthick = float()
        self.slits = list()
        self.slitpoints = np.array([])
        self.slitTree = list()

        if len(args) > 0 and type(args[0]).__name__ in ['Trajectory_human', 'Trajectory_ps_simulation',
                                                        'Trajectory_ant', 'Trajectory_gillespie', 'Trajectory',
                                                        'Trajectory_part', 'Trajectory_humanhand']:
            x = args[0]
            if geometry is None:
                self.excel_file_maze, self.excel_file_load = x.geometry()
            elif x.solver == 'gillespie':
                self.excel_file_maze, self.excel_file_load = \
                    ('MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx',
                     'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx')
            else:
                self.excel_file_maze, self.excel_file_load = geometry
            self.shape = x.shape
            self.size = x.size
            self.solver = x.solver
            position = x.position[i] if position is None else position
            angle = x.angle[i] if angle is None else angle
        else:
            self.excel_file_maze, self.excel_file_load = geometry
            self.shape = shape
            self.size = size
            self.solver = solver

        is_exp_valid(self.shape, self.solver, self.size)
        self.getMazeDim()
        super().__init__(position=position, angle=angle, point_particle=point_particle, bb=bb)
        self.CreateSlitObject()

    def corners(self):
        corners = [[0, 0],
                   [0, self.arena_height],
                   [self.slits[-1] + 20, self.arena_height],
                   [self.slits[-1] + 20, 0],
                   ]
        return np.array(corners + list(np.resize(self.slitpoints, (16, 2))))

    def getMazeDim(self):

        df = read_excel(path.join(maze_dimension_directory, self.excel_file_maze), engine='openpyxl')

        if self.excel_file_maze in ['MazeDimensions_ant.xlsx', 'MazeDimensions_ant_L_I_425.xlsx',
                                    'MazeDimensions_new2021_SPT_ant.xlsx',
                                    'MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx']:  # all measurements in cm
            d = df.loc[df['Name'] == self.size + '_' + self.shape]
            self.arena_length = d['arena_length'].values[0]
            self.arena_height = d['arena_height'].values[0]
            self.exit_size = d['exit_size'].values[0]
            self.wallthick = d['wallthick'].values[0]
            if type(d['slits'].values[0]) == str:
                self.slits = [[float(s) for s in d['slits'].values[0].split(', ')][0],
                              [float(s) for s in d['slits'].values[0].split(', ')][1]]
            else:
                self.slits = [d['slits'].values[0]]

        elif self.excel_file_maze in ['MazeDimensions_humanhand.xlsx']:  # only SPT
            d = df.loc[df['Name'] == 'humanhand']
            self.arena_length = d['arena_length'].values[0]
            self.arena_height = d['arena_height'].values[0]
            self.exit_size = d['exit_size'].values[0]
            self.wallthick = d['wallthick'].values[0]
            self.slits = [float(s) for s in d['slits'].values[0].split(', ')]

        elif self.excel_file_maze in ['MazeDimensions_human.xlsx']:  # all measurements in meters
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

        self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)

    def in_what_Chamber(self, r: np.array) -> Union[int, None]:
        """
        Only works for SPT maze
        Check the x coordinate of the object and return the chamber number
        __________________
                  |    |
            0       1    2
                  |    |
        __________________
        :param r: position of the particle
        :return: the index of the chamber in which the particle is
        """
        if self.shape != 'SPT':
            raise ValueError('No Chambers defined for shape ' + self.shape)

        if r[0] < self.slits[0]:
            return 0
        elif r[0] < self.slits[1]:
            return 1
        elif r[0] < self.slits[1] + (self.slits[1]-self.slits[0]):
            return 2
        return None

    def CreateSlitObject(self):
        # # The x and y position describe the point, where the middle (in x direction) of the top edge (y direction)
        # of the lower wall of the slit is...
        if self.shape == 'LongT':
            # TODO
            pass

        # We need a special case for L_SPT because in the manufacturing the slits were not vertically glued
        if self.size == 'L' and self.shape == 'SPT' and self.excel_file_maze == 'MazeDimensions_ant_old.xlsx':
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
        else:
            self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)
            for i, slit in enumerate(self.slits):
                # this is the lower Slit
                self.slitpoints[2 * i] = np.array([[slit, 0],
                                                   [slit, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, 0]]
                                                  )

                self.maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i].tolist())

                # this is the upper Slit
                self.slitpoints[2 * i + 1] = np.array([[slit, (self.arena_height + self.exit_size) / 2],
                                                       [slit, self.arena_height],
                                                       [slit + self.wallthick, self.arena_height],
                                                       [slit + self.wallthick,
                                                        (self.arena_height + self.exit_size) / 2]]
                                                      )

                self.maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i + 1].tolist())

            # I dont want to have the vertical line at the first exit
            self.slitTree = BoxIt(np.array([[0, 0],
                                            [0, self.arena_height],
                                            [self.slits[-1], self.arena_height],
                                            [self.slits[-1], 0]]),
                                  0.1, without='right')

            for slit_points in self.slitpoints:
                self.slitTree = np.vstack((self.slitTree, BoxIt(slit_points, 0.01)))

            self.slitTree = cKDTree(self.slitTree)

    # def get_zone(self):
    #     if self.shape == 'SPT':
    #         self.zone = np.array([[0, 0],
    #                               [0, self.arena_height],
    #                               [self.slits[0], self.arena_height],
    #                               [self.slits[0], 0]])
    #     else:
    #         RF = ResizeFactors[self.solver][self.size]
    #         self.zone = np.array(
    #             [[self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 - self.arena_height * RF / 2],
    #              [self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 + self.arena_height * RF / 2],
    #              [self.slits[0], self.arena_height / 2 + self.arena_height * RF / 2],
    #              [self.slits[0], self.arena_height / 2 - self.arena_height * RF / 2]])
    #     return

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

    def minimal_path_length(self):
        from DataFrame.dataFrame import myDataFrame
        # from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
        p = myDataFrame.loc[myDataFrame['filename'] == filename_dstar(self.size, self.shape, 0, 0)][
            ['path length [length unit]']]
        return p.values[0][0]

    def create_Maze(self):
        my_maze = self.CreateBody(b2BodyDef(position=(0, 0), angle=0, type=b2_staticBody, userData='maze'))
        my_maze.CreateLoopFixture(
            vertices=[(0, 0), (0, float(self.arena_height)), (float(self.arena_length), float(self.arena_height)),
                      (float(self.arena_length), 0)])
        return my_maze


class Maze_free_space(Maze_parent):
    def __init__(self, *args, size='XL', shape='SPT', solver='ant', position=None, angle=0, point_particle=False,
                 geometry: tuple = None, i=0, bb: bool = False):
        if len(args) > 0 and type(args[0]).__name__ in ['Trajectory_human', 'Trajectory_ps_simulation',
                                                        'Trajectory_ant', 'TrajectoryGillespie', 'Trajectory',
                                                        'Trajectory_part']:
            x = args[0]
            self.arena_height = np.max(x.position[:, 1])
            self.arena_length = np.max(x.position[:, 0])
            self.excel_file_maze, self.excel_file_load = x.geometry()
            self.shape = x.shape
            self.size = x.size
            self.solver = x.solver
            position = x.position[i] if position is None else position
            angle = x.angle[i] if angle is None else angle
        else:
            self.arena_height = 10
            self.arena_length = 10
            self.excel_file_maze, self.excel_file_load = geometry
            self.shape = shape
            self.size = size
            self.solver = solver

        super().__init__(position=position, angle=angle, point_particle=point_particle, bb=bb, free=True)

    def create_Maze(self):
        my_maze = self.CreateBody(b2BodyDef(position=(0, 0), angle=0, type=b2_staticBody, userData='maze'))
        my_maze.CreateLoopFixture(
            vertices=[(0, 0), (0, self.arena_height), (self.arena_length, self.arena_height),
                      (self.arena_length, 0)])
        return my_maze

#
# print({size: Maze(size=size, shape='SPT', solver='human',
#                   geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')).average_radius()
#        for size in ['Large', 'Medium', 'Small Far']})
