import numpy as np
from Box2D import b2BodyDef, b2_dynamicBody, b2Vec2, b2CircleShape, b2FixtureDef
from Setup.Maze import ResizeFactors

periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}
shift = - 0.10880829015544041
assymetric_h_shift = 1.22 * 2


# I multiply all these values with 2, because I got them in L, but want to state them in XL.
# StartedScripts: Does the load have a preferred orientation while moving?


def Loops(Box2D_Object, vertices=None):
    if vertices is None:
        vertices = []

    if hasattr(Box2D_Object, 'bodies'):
        for body in Box2D_Object.bodies:
            Loops(body, vertices=vertices)
    else:
        for fixture in Box2D_Object.fixtures:  # Here, we update the vertices of our bodies.fixtures and...
                vertices.append(
                    [(Box2D_Object.transform * v) for v in fixture.shape.vertices][:4])  # Save vertices of the load
    return vertices


def sites(Box2D_Object, sites, gillespie):
    """
    :param Box2D_Object: Object, usually a b2Body, that contains fixtures
    :param sites: how many attachment sites do you want to distribute around your shape
    :return: a np.array of the form [[x0, y0], ..., [x_(sites-1), y_(sites-1)]], with the attachment positions in the coordinate
    system of the load. This is specifically built to assist the Gillespie Code
    """

    if hasattr(Box2D_Object, 'bodies'):
        for body in Box2D_Object.bodies:
            return sites(body)

    points = np.zeros([0, 2])
    for fixture in Box2D_Object.fixtures:  # Here, we update the vertices of our bodies.fixtures and...
        if isinstance(fixture.shape, b2CircleShape):
            [gillespie.ant_vector(i) for i in range(len(gillespie.n_p))]
            array = np.zeros([sites, 2])
            points = np.vstack([points, array])
        else:
            # TODO: implement for polygon
            pass
    return points


def average_radius(size, shape, solver):
    r = ResizeFactors[solver][size]
    SPT_radius = 0.76791  # you have to multiply this with the shape width to get the average radius
    radii = dict()
    if solver in ['ant', 'dstar']:
        radii = {'H': 2.9939 * r,
                 'I': 2.3292 * r,
                 'T': 2.9547 * r,
                 # 'SPT': 2 * 3.7052 * r, historical...
                 'SPT': SPT_radius * getLoadDim('ant', 'SPT', 'XL')[1] * r,
                 'RASH': 2 * 1.6671 * r,
                 'LASH': 2 * 1.6671 * r}
    elif solver == 'human':
        radii = {'SPT': SPT_radius * getLoadDim('human', 'SPT', size)[1]}
    elif solver == 'humanhand':
        radii = {'SPT': SPT_radius * getLoadDim('humanhand', 'SPT', size)[1]}

    return radii[shape]


def getLoadDim(solver, shape, size):
    if solver in ['ant', 'dstar', 'sim']:
        resize_factor = ResizeFactors[solver][size]
        shape_sizes = {'H': [5.6, 7.2, 1.6],
                       'SPT': [4.85, 9.65, 0.85, 2.44],
                       'LASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                       'RASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                       'I': [5.5, 1.75, 1.75],
                       'T': [5.4, 5.6, 1.6]
                       }
        # dimensions = [shape_height, shape_width, shape_thickness, long_edge/short_edge]
        dimensions = [i * resize_factor for i in shape_sizes[shape]]

        if (resize_factor == 1) and shape[1:] == 'ASH':  # for XL ASH
            dimensions = [le * resize_factor for le in [8.14, 5.6, 1.2]]
        elif (resize_factor == 0.75) and shape[1:] == 'ASH':  # for XL ASH
            dimensions = [le * resize_factor for le in [9, 6.2, 1.2]]
        return dimensions

    elif solver == 'human':
        # [shape_height, shape_width, shape_thickness, short_edge]
        SPT_Human_sizes = {'S': [0.805, 1.61, 0.125, 0.805 / 0.405],
                           'M': [1.59, 3.18, 0.240, 1.59 / 0.795],
                           'L': [3.2, 6.38, 0.51, 3.2 / 1.585]}
        return SPT_Human_sizes[size[0]]

    elif solver == 'humanhand':
        # [shape_height, shape_width, shape_thickness, short_edge]
        SPT_Human_sizes = [6, 12.9, 0.9, 3]
        return SPT_Human_sizes


def AddLoadFixtures(load, size, shape, solver):
    assymetric_h_shift = 1.22 * 2

    if shape == 'circle':
        load.CreateFixture(b2FixtureDef(shape=b2CircleShape(pos=(0, 0), radius=0.2)),
                           density=1, friction=0, restitution=0,
                           )

    if shape == 'H':
        [shape_height, shape_width, shape_thickness] = getLoadDim(solver, shape, size)
        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, shape_thickness / 2),
            (shape_width / 2, -shape_thickness / 2),
            (-shape_width / 2, -shape_thickness / 2),
            (-shape_width / 2, shape_thickness / 2)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, -shape_height / 2),
            (shape_width / 2, shape_height / 2),
            (shape_width / 2 - shape_thickness, shape_height / 2),
            (shape_width / 2 - shape_thickness, -shape_height / 2)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (-shape_width / 2, -shape_height / 2),
            (-shape_width / 2, shape_height / 2),
            (-shape_width / 2 + shape_thickness, shape_height / 2),
            (-shape_width / 2 + shape_thickness, -shape_height / 2)],
            density=1, friction=0, restitution=0,
        )

        load.corners = np.array([[shape_width / 2, -shape_height / 2],
                                 [-shape_width / 2, -shape_height / 2],
                                 [shape_width / 2, shape_height / 2],
                                 [-shape_width / 2, shape_height / 2]])

    if shape == 'I':
        [shape_height, _, shape_thickness] = getLoadDim(solver, shape, size)
        load.CreatePolygonFixture(vertices=[
            (shape_height / 2, -shape_thickness / 2),
            (shape_height / 2, shape_thickness / 2),
            (-shape_height / 2, shape_thickness / 2),
            (-shape_height / 2, -shape_thickness / 2)],
            density=1, friction=0, restitution=0,
        )
        load.corners = np.array([[shape_height / 2, -shape_thickness / 2],
                                 [-shape_height / 2, -shape_thickness / 2],
                                 [shape_height / 2, shape_thickness / 2],
                                 [-shape_height / 2, shape_thickness / 2]])

    if shape == 'T':
        [shape_height, shape_width, shape_thickness] = getLoadDim(solver, shape, size)
        resize_factor = ResizeFactors[solver][size]
        h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower part of the T.

        #  Top horizontal T part
        load.CreatePolygonFixture(vertices=[
            (-((shape_height - shape_thickness) / 2) + h, -shape_width / 2),
            (-((shape_height + shape_thickness) / 2) + h, -shape_width / 2),
            (-((shape_height + shape_thickness) / 2) + h, shape_width / 2),
            (-((shape_height - shape_thickness) / 2) + h, shape_width / 2)],
            density=1, friction=0, restitution=0,
        )

        #  Bottom vertical T part
        load.CreatePolygonFixture(vertices=[
            (-(shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
            ((shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
            ((shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
            (-(shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
        ],
            density=1, friction=0, restitution=0,
        )

        load.corners = np.array([[-((shape_height + shape_thickness) / 2) + h, -shape_width / 2],
                                 [-((shape_height + shape_thickness) / 2) + h, shape_width / 2],
                                 [-(shape_height - shape_thickness) / 2 + h, -shape_thickness / 2],
                                 [-(shape_height - shape_thickness) / 2 + h, shape_thickness / 2]])

    '''        corners = np.array([[shape_thickness/2, -shape_height/2-h],
                                [-shape_thickness/2, -shape_height/2-h],
                                [shape_width/2, shape_height/2-h],
                                [-shape_width/2, shape_height/2-h]])
        
        '''

    if shape == 'SPT':  # This is the Special T
        [shape_height, shape_width, shape_thickness, _] = getLoadDim(solver, shape, size)
        print(str(getLoadDim(solver, shape, size)))

        # h = SPT_centroid_shift * ResizeFactors[x.size]  # distance of the centroid away from the center of the long middle
        h = shift * shape_width  # distance of the centroid away from the center of the long middle
        # part of the T. (1.445 calculated)

        # This is the connecting middle piece
        load.CreatePolygonFixture(vertices=[
            (shape_width / 2 - h, shape_thickness / 2),
            (shape_width / 2 - h, -shape_thickness / 2),
            (-shape_width / 2 - h, -shape_thickness / 2),
            (-shape_width / 2 - h, shape_thickness / 2)],
            density=1, friction=0, restitution=0,
        )

        # This is the short side
        load.CreatePolygonFixture(vertices=[
            (shape_width / 2 - h, -shape_height / 2 * 2.44 / 4.82),
            # This addition is because the special T looks like an H where one vertical side is shorter by a factor
            # 2.44/4.82
            (shape_width / 2 - h, shape_height / 2 * 2.44 / 4.82),
            (shape_width / 2 - shape_thickness - h, shape_height / 2 * 2.44 / 4.82),
            (shape_width / 2 - shape_thickness - h, -shape_height / 2 * 2.44 / 4.82)],
            density=1, friction=0, restitution=0,
        )

        # This is the long side
        load.CreatePolygonFixture(vertices=[
            (-shape_width / 2 - h, -shape_height / 2),
            (-shape_width / 2 - h, shape_height / 2),
            (-shape_width / 2 + shape_thickness - h, shape_height / 2),
            (-shape_width / 2 + shape_thickness - h, -shape_height / 2)],
            density=1, friction=0, restitution=0,
        )

        load.corners = np.array([[shape_width / 2, -shape_height / 2 * 2.44 / 4.82],
                                 [-shape_width / 2, -shape_height / 2],
                                 [shape_width / 2, shape_height / 2 * 2.44 / 4.82],
                                 [-shape_width / 2, shape_height / 2]])

    if shape == 'RASH':  # This is the ASymmetrical H
        [shape_height, shape_width, shape_thickness] = getLoadDim(solver, shape, size)
        assymetric_h_shift = assymetric_h_shift * ResizeFactors[solver][size]
        # I multiply all these values with 2, because I got them in L, but want to state
        # them in XL.
        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, shape_thickness / 2,),
            (shape_width / 2, -shape_thickness / 2,),
            (-shape_width / 2, -shape_thickness / 2,),
            (-shape_width / 2, shape_thickness / 2,)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
            # This addition is because the special T looks like an H where one vertical side is shorter by a factor
            # 2.44/4.82
            (shape_width / 2, shape_height / 2,),
            (shape_width / 2 - shape_thickness, shape_height / 2,),
            (shape_width / 2 - shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (-shape_width / 2, -shape_height / 2,),
            (-shape_width / 2, shape_height / 2 - assymetric_h_shift,),
            (-shape_width / 2 + shape_thickness, shape_height / 2 - assymetric_h_shift,),
            (-shape_width / 2 + shape_thickness, -shape_height / 2,)],
            density=1, friction=0, restitution=0,
        )

    if shape == 'LASH':  # This is the ASymmetrical H
        [shape_height, shape_width, shape_thickness] = getLoadDim(solver, shape, size)
        assymetric_h_shift = assymetric_h_shift * ResizeFactors[solver][size]
        # I multiply all these values with 2, because I got them in L, but want to state
        # them in XL.
        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, shape_thickness / 2,),
            (shape_width / 2, -shape_thickness / 2,),
            (-shape_width / 2, -shape_thickness / 2,),
            (-shape_width / 2, shape_thickness / 2,)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (shape_width / 2, -shape_height / 2,),
            # This addition is because the special T looks like an H where one vertical side is shorter by a factor
            # 2.44/4.82
            (shape_width / 2, shape_height / 2 - assymetric_h_shift,),
            (shape_width / 2 - shape_thickness, shape_height / 2 - assymetric_h_shift,),
            (shape_width / 2 - shape_thickness, -shape_height / 2,)],
            density=1, friction=0, restitution=0,
        )

        load.CreatePolygonFixture(vertices=[
            (-shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
            (-shape_width / 2, shape_height / 2,),
            (-shape_width / 2 + shape_thickness, shape_height / 2,),
            (-shape_width / 2 + shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
            density=1, friction=0, restitution=0,
        )

        # load.corners = np.array([[ -shape_height/2,shape_width/2, ],
        #                     [ -shape_height/2+assymetric_h_shift, -shape_width/2,],
        #                     [ shape_height/2-assymetric_h_shift, shape_width/2,],
        #                     [ shape_height/2,-shape_width/2, ]])

    return load


def circumference(x):
    from Setup.Maze import ResizeFactors
    shape_thickness, shape_height, shape_width = getLoadDim(x.solver, x.shape,
                                                            ResizeFactors[x.size])
    if x.shape.endswith('ASH'):
        print('I dont know circumference of ASH!!!')
        breakpoint()
    cir = {'H': 4 * shape_height - 2 * shape_thickness + 2 * shape_width,
           'I': 2 * shape_height + 2 * shape_width,
           'T': 2 * shape_height + 2 * shape_width,
           'SPT': 2 * shape_height / 2 * 2.44 / 4.82 + 2 * shape_height - 2 * shape_thickness + 2 * shape_width,
           'RASH': 2 * shape_width + 4 * shape_height - 4 * assymetric_h_shift * ResizeFactors[
               x.size] - 2 * shape_thickness,
           'LASH': 2 * shape_width + 4 * shape_height - 4 * assymetric_h_shift * ResizeFactors[
               x.size] - 2 * shape_thickness
           }
    return cir[x.shape]


def Load(my_maze, position=None, angle=0, point_particle=False):
    if position is None:
        position = [0, 0]
    my_load = my_maze.CreateBody(b2BodyDef(position=(float(position[0]), float(position[1])),
                                           angle=float(angle),
                                           type=b2_dynamicBody,
                                           fixedRotation=False,
                                           linearDamping=0,
                                           angularDamping=0),
                                 restitution=0,
                                 friction=0,
                                 )

    my_load.userData = 'my_load'
    if not point_particle:
        my_load = AddLoadFixtures(my_load, my_maze.size, my_maze.shape, my_maze.solver)
    return my_load


def force_attachment_positions(my_load, x):
    from Classes_Experiment.humans import participant_number
    if x.solver == 'human' and x.size == 'Medium' and x.shape == 'SPT':

        # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)
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
        h = shift * shape_width

    elif x.solver == 'human' and x.size == 'Large' and x.shape == 'SPT':
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)

        xMNOP = -shape_width / 2,
        xLQ = xMNOP + shape_thickness / 2
        xAB = (-1) * xMNOP
        xCZ = (-1) * xLQ
        xKR = xMNOP + shape_thickness
        xDY, xEX, xFW, xGV, xHU, xIT, xJS = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

        yA_B = short_edge / 6
        yC_Z = short_edge / 2
        yDEFGHIJ_STUVWXY = shape_thickness / 2
        yK_R = shape_height / 10 * 2
        yL_Q = shape_height / 2
        yM_P = shape_height / 10 * 3
        yN_O = shape_height / 10

        positions = [[xAB, yA_B],
                     [xAB, - yA_B],
                     [xCZ, yC_Z],
                     [xDY, yDEFGHIJ_STUVWXY],
                     [xEX, yDEFGHIJ_STUVWXY],
                     [xFW, yDEFGHIJ_STUVWXY],
                     [xGV, yDEFGHIJ_STUVWXY],
                     [xHU, yDEFGHIJ_STUVWXY],
                     [xIT, yDEFGHIJ_STUVWXY],
                     [xJS, yDEFGHIJ_STUVWXY],
                     [xKR, yK_R],
                     [xLQ, yL_Q],
                     [xMNOP, yM_P],
                     [xMNOP, yN_O],
                     [xMNOP, -yN_O],
                     [xMNOP, -yM_P],
                     [xLQ, -yL_Q],
                     [xKR, -yK_R],
                     [xJS, -yDEFGHIJ_STUVWXY],
                     [xIT, -yDEFGHIJ_STUVWXY],
                     [xHU, -yDEFGHIJ_STUVWXY],
                     [xGV, -yDEFGHIJ_STUVWXY],
                     [xFW, -yDEFGHIJ_STUVWXY],
                     [xEX, -yDEFGHIJ_STUVWXY],
                     [xDY, -yDEFGHIJ_STUVWXY],
                     [xCZ, -yC_Z],
                     ]
        h = shift * shape_width

    else:
        positions = [[0, 0] for i in range(participant_number[x.size])]
        h = 0

    # shift the shape...
    positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame

    #
    return [my_load.GetWorldPoint(b2Vec2(r)) for r in positions]  # r vectors in the lab frame
