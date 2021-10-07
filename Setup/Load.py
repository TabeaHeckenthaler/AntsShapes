import numpy as np
from Box2D import b2BodyDef, b2_dynamicBody, b2Vec2, b2CircleShape, b2FixtureDef
from Setup.Maze import ResizeFactors

ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie']  # also in Maze.py

periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}
assymetric_h_shift = 1.22 * 2

# somehow these contain the same information
SPT_ratio = 2.44 / 4.82
centerOfMass_shift = - 0.10880829015544041


# I multiply all these values with 2, because I got them in L, but want to state them in XL.
# StartedScripts: Does the load have a preferred orientation while moving?


def Loops(Box2D_Object, vertices=None):
    if vertices is None:
        vertices = []

    if hasattr(Box2D_Object, 'bodies'):
        for body in Box2D_Object.bodies:
            Loops(body, vertices=vertices)
    else:
        for fixture in Box2D_Object.fixtures:  # Here, we update_screen the vertices of our bodies.fixtures and...
            vertices.append(
                [(Box2D_Object.transform * v) for v in fixture.shape.vertices][:4])  # Save vertices of the load
    return vertices


def average_radius(size, shape, solver):
    r = ResizeFactors[solver][size]
    SPT_radius = 0.76791  # you have to multiply this with the shape width to get the average radius
    radii = dict()
    if solver in ['ant', 'ps_simulation']:
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


def getLoadDim(solver: str, shape: str, size: str, short_edge=False):
    """

    """
    if solver in ant_dimensions:
        resize_factor = ResizeFactors[solver][size]
        shape_sizes = {'H': [5.6, 7.2, 1.6],
                       'SPT': [4.85, 9.65, 0.85],
                       'LASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                       'RASH': [5.24 * 2, 3.58 * 2, 0.8 * 2],
                       'I': [5.5, 1.75, 1.75],
                       'T': [5.4, 5.6, 1.6]
                       }
        if short_edge:
            shape_sizes['SPT'] = [4.85, 9.65, 0.85, 4.85*SPT_ratio]
        # dimensions = [shape_height, shape_width, shape_thickness, optional: long_edge/short_edge]
        dimensions = [i * resize_factor for i in shape_sizes[shape]]

        if (resize_factor == 1) and shape[1:] == 'ASH':  # for XL ASH
            dimensions = [le * resize_factor for le in [8.14, 5.6, 1.2]]
        elif (resize_factor == 0.75) and shape[1:] == 'ASH':  # for XL ASH
            dimensions = [le * resize_factor for le in [9, 6.2, 1.2]]
        return dimensions

    elif solver == 'human':
        # [shape_height, shape_width, shape_thickness, short_edge]
        if short_edge:
            SPT_Human_sizes = {'S': [0.805, 1.61, 0.125, 0.405],
                               'M': [1.59, 3.18, 0.240, 0.795],
                               'L': [3.2, 6.38, 0.51, 1.585]}
        else:
            SPT_Human_sizes = {'S': [0.805, 1.61, 0.125],
                               'M': [1.59, 3.18, 0.240],
                               'L': [3.2, 6.38, 0.51]}
        return SPT_Human_sizes[size[0]]

    elif solver == 'humanhand':
        # [shape_height, shape_width, shape_thickness, short_edge]
        SPT_Human_sizes = [6, 12.9, 0.9, 3]
        if not short_edge:
            SPT_Human_sizes = SPT_Human_sizes[:3]
        return SPT_Human_sizes


def AddLoadFixtures(load, size, shape, solver):
    assymetric_h_shift = 1.22 * 2

    if shape == 'circle':
        from trajectory_inheritance.gillespie import radius
        load.CreateFixture(b2FixtureDef(shape=b2CircleShape(pos=(0, 0), radius=radius)),
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

        load.corners = np.array([[-shape_width / 2 + shape_thickness, -shape_thickness / 2],
                                 [-shape_width / 2 + shape_thickness, -shape_height / 2],
                                 [-shape_width / 2, -shape_height / 2],
                                 [-shape_width / 2, shape_height / 2],
                                 [-shape_width / 2 + shape_thickness, shape_height / 2],
                                 [-shape_width / 2 + shape_thickness, shape_thickness / 2],
                                 [shape_width / 2 - shape_thickness, shape_thickness / 2],
                                 [shape_width / 2 - shape_thickness, shape_height / 2],
                                 [shape_width / 2, shape_height / 2],
                                 [shape_width / 2, -shape_height / 2],
                                 [shape_width / 2 - shape_thickness, -shape_height / 2],
                                 [shape_width / 2 - shape_thickness, -shape_thickness / 2]])

        load.phis = np.array([0, -np.pi / 2, np.pi, np.pi / 2, 0, np.pi/2,
                              np.pi, np.pi/2, 0, -np.pi/2, np.pi, -np.pi/2])

    if shape == 'I':
        [shape_height, _, shape_thickness] = getLoadDim(solver, shape, size)
        load.CreatePolygonFixture(vertices=[
            (shape_height / 2, -shape_thickness / 2),
            (shape_height / 2, shape_thickness / 2),
            (-shape_height / 2, shape_thickness / 2),
            (-shape_height / 2, -shape_thickness / 2)],
            density=1, friction=0, restitution=0,
        )

        # old version
        # load.corners = np.array([[shape_height / 2, -shape_thickness / 2],
        #                          [-shape_height / 2, -shape_thickness / 2],
        #                          [shape_height / 2, shape_thickness / 2],
        #                          [-shape_height / 2, shape_thickness / 2]])

        load.corners = np.array([[-shape_height / 2, -shape_thickness / 2],
                                 [-shape_height / 2, shape_thickness / 2],
                                 [shape_height / 2, shape_thickness / 2],
                                 [shape_height / 2, -shape_thickness / 2]])

        # phis describe the angles of the normal between the corners to the x axis of the world coordinates
        load.phis = np.array([np.pi, np.pi/2, 0, -np.pi/2])

    if shape == 'T':
        [shape_height, shape_width, shape_thickness] = getLoadDim(solver, shape, size)
        resize_factor = ResizeFactors[solver][size]
        h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower part of the T.

        #  Top horizontal T part
        load.CreatePolygonFixture(vertices=[
            ((-shape_height + shape_thickness) / 2 + h, -shape_width / 2),
            ((-shape_height - shape_thickness) / 2 + h, -shape_width / 2),
            ((-shape_height - shape_thickness) / 2 + h, shape_width / 2),
            ((-shape_height + shape_thickness) / 2 + h, shape_width / 2)],
            density=1, friction=0, restitution=0,
        )

        #  Bottom vertical T part
        load.CreatePolygonFixture(vertices=[
            ((-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2),
            ((shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
            ((shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
            ((-shape_height + shape_thickness) / 2 + h, shape_thickness / 2),
        ],
            density=1, friction=0, restitution=0,
        )

        # load.corners = np.array([[-((shape_height + shape_thickness) / 2) + h, -shape_width / 2],
        #                          [-((shape_height + shape_thickness) / 2) + h, shape_width / 2],
        #                          [-(shape_height - shape_thickness) / 2 + h, -shape_thickness / 2],
        #                          [-(shape_height - shape_thickness) / 2 + h, shape_thickness / 2]])

        # the corners  in my_load.corners must be ordered like this: finding the intersection of the negative y-axis,
        # and the shape, and going clockwise find the first corner. Then go clockwise in order of the corners.
        # TODO: implement corners and phis
        load.corners = np.array([[(shape_height - shape_thickness) / 2 + h, -shape_thickness / 2],
                                 [(-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2],
                                 [(-shape_height + shape_thickness) / 2 + h, -shape_width / 2],
                                 [(-shape_height - shape_thickness) / 2 + h, -shape_width / 2],
                                 [(-shape_height - shape_thickness) / 2 + h, shape_width / 2],
                                 [(-shape_height + shape_thickness) / 2 + h, shape_width / 2],
                                 [(-shape_height + shape_thickness) / 2 + h, shape_thickness / 2],
                                 [(shape_height - shape_thickness) / 2 + h, shape_thickness / 2]])

        # phis describe the angles of the normal between the corners to the x axis of the world coordinates.
        # Starting at first corner of load.corners, and going clockwise
        # load.phis = np.array([np.pi, np.pi / 2, 0, -np.pi / 2])
        load.phis = np.array([np.pi, -np.pi / 2, np.pi, np.pi/2, 0, -np.pi/2, 0, -np.pi/2])

    if shape == 'SPT':  # This is the Special T
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(solver, shape, size, short_edge=True)

        # h = SPT_centroid_shift * ResizeFactors[x.size]  # distance of the centroid away from the center of the long middle
        h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
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
            (shape_width / 2 - h, -short_edge / 2),
            # This addition is because the special T looks like an H where one vertical side is shorter by a factor
            # SPT_ratio
            (shape_width / 2 - h, short_edge / 2),
            (shape_width / 2 - shape_thickness - h, short_edge / 2),
            (shape_width / 2 - shape_thickness - h, -short_edge / 2)],
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

        # load.corners = np.array([[shape_width / 2, -shape_height / 2 * SPT_ratio],
        #                          [-shape_width / 2, -shape_height / 2],
        #                          [shape_width / 2, shape_height / 2 * SPT_ratio],
        #                          [-shape_width / 2, shape_height / 2]])

        # the corners  in my_load.corners must be ordered like this: finding the intersection of the negative y-axis,
        # and the shape, and going clockwise find the first corner. Then go clockwise in order of the corners.
        # corners = np.vstack([[0, my_load.corners[0, 1]], my_load.corners])
        # phis = np.append(my_load.phis, my_load.phis[0])
        load.corners = np.array([[-shape_width / 2 + shape_thickness - h, -shape_thickness / 2],  # left
                                 [-shape_width / 2 + shape_thickness - h, -shape_height / 2],
                                 [-shape_width / 2 - h, -shape_height / 2],
                                 [-shape_width / 2 - h, shape_height / 2],
                                 [-shape_width / 2 + shape_thickness - h, shape_height / 2],
                                 [-shape_width / 2 + shape_thickness - h, shape_thickness / 2],
                                 [shape_width / 2 - shape_thickness - h, shape_thickness / 2],  # right
                                 [shape_width / 2 - shape_thickness - h, shape_height / 2 * SPT_ratio],
                                 [shape_width / 2 - h, shape_height / 2 * SPT_ratio],
                                 [shape_width / 2 - h, -shape_height / 2 * SPT_ratio],
                                 [shape_width / 2 -shape_thickness - h, -shape_height / 2 * SPT_ratio],
                                 [shape_width / 2 -shape_thickness - h, -shape_thickness / 2]])

        # phis describe the angles of the normal between the corners to the x axis of the world coordinates.
        # Starting at first corner of load.corners, and going clockwise
        # load.phis = np.array([np.pi, np.pi / 2, 0, -np.pi / 2])
        load.phis = np.array([0, -np.pi / 2, np.pi, np.pi / 2, 0, np.pi / 2,
                              np.pi, np.pi / 2, 0, -np.pi / 2, np.pi, np.pi / 2])

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
            # SPT_ratio
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
    # TODO: implement corners and phis

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
            # SPT_ratio
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

        # TODO: implement corners and phis
        # load.corners = np.array([[ -shape_height/2,shape_width/2, ],
        #                     [ -shape_height/2+assymetric_h_shift, -shape_width/2,],
        #                     [ shape_height/2-assymetric_h_shift, shape_width/2,],
        #                     [ shape_height/2,-shape_width/2, ]])

    return load


def circumference(solver, shape, size):
    from Setup.Maze import ResizeFactors
    shape_height, shape_width, shape_thickness = getLoadDim(solver, shape, size)

    if shape.endswith('ASH'):
        print('I dont know circumference of ASH!!!')
        breakpoint()
    cir = {'H': 4 * shape_height - 2 * shape_thickness + 2 * shape_width,
           'I': 2 * shape_height + 2 * shape_width,
           'T': 2 * shape_height + 2 * shape_width,
           # 'SPT': 2 * shape_height / 2 * SPT_ratio +
           #        2 * shape_height -
           #        2 * shape_thickness +
           #        2 * shape_width,
           'SPT': 2 * shape_height * SPT_ratio +
                  2 * shape_height -
                  2 * shape_thickness +
                  2 * shape_width,
           'RASH': 2 * shape_width + 4 * shape_height - 4 * assymetric_h_shift * ResizeFactors[solver][size]
                   - 2 * shape_thickness,
           'LASH': 2 * shape_width + 4 * shape_height - 4 * assymetric_h_shift * ResizeFactors[solver][size]
                   - 2 * shape_thickness
           }
    return cir[shape]


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
    my_load.size = my_maze.size
    my_load.shape = my_maze.shape
    my_load.solver = my_maze.solver

    if not point_particle:
        my_load = AddLoadFixtures(my_load, my_maze.size, my_maze.shape, my_maze.solver)
    return my_load


def Gillespie_sites_angels(my_load, n: int):
    """

    :param my_load: b2Body
    :param n: number of attachment sites
    :param x: trajectory_inheritance object, if we need to know about size, etc.
    :return: 2xn numpy matrix, with x and y coordinates of attachment sites, in the my_load coordinate system 
    and n numpy matrix, with angles of normals of attachment sites, measured against the load coordinate system
    """
    if my_load.shape == 'circle':
        from trajectory_inheritance.gillespie import radius
        theta = -np.linspace(0, 2 * np.pi, n)
        sites = radius * np.transpose(np.vstack([np.cos(theta), np.sin(theta)]))
        phi_default = theta
        return sites, phi_default

    else:
        # the corners  in my_load.corners must be ordered like this: finding the intersection of the negative y-axis,
        # and the shape, and going clockwise find the first corner. Then go clockwise in order of the corners.
        # corners = np.vstack([[0, my_load.corners[0, 1]], my_load.corners])
        # phis = np.append(my_load.phis, my_load.phis[0])

        def linear_combination(step_size, start, end):
            return start + step_size * (end - start) / np.linalg.norm(start - end)

        # walk around the shape
        i = 1
        delta = circumference(my_load.solver, my_load.shape, my_load.size) / n
        step_size = delta
        sites = np.array([linear_combination(0.5, my_load.corners[0], my_load.corners[1])])
        phi_default = np.array([my_load.phis[0]])
        start = sites[-1]
        aim = my_load.corners[1]

        while sites.shape[0] < n:
            if np.linalg.norm(start - aim) > step_size:
                sites = np.vstack([sites, linear_combination(step_size, start, aim)])
                start = sites[-1]
                phi_default = np.append(phi_default, my_load.phis[(i-1) % my_load.corners.shape[0]])
                step_size = delta

            else:
                step_size = step_size - np.linalg.norm(start - aim)
                i = i + 1
                start = my_load.corners[(i-1) % my_load.corners.shape[0]]
                aim = my_load.corners[i % my_load.corners.shape[0]]
        return sites, phi_default


def force_attachment_positions(my_load, x):
    from trajectory_inheritance.humans import participant_number
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
        h = centerOfMass_shift * shape_width

    elif x.solver == 'human' and x.size == 'Large' and x.shape == 'SPT':
        # [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size)
        [shape_height, shape_width, shape_thickness, short_edge] = getLoadDim(x.solver, x.shape, x.size,
                                                                              short_edge=True)

        xMNOP = -shape_width / 2
        xLQ = xMNOP + shape_thickness / 2
        xAB = (-1) * xMNOP
        xCZ = (-1) * xLQ
        xKR = xMNOP + shape_thickness
        # xDY, xEX, xFW, xGV, xHU, xIT, xJS = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]
        xJS, xIT, xHU, xGV, xFW, xEX, xDY = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

        yA_B = short_edge / 6
        yC_Z = short_edge / 2
        yDEFGHIJ_STUVWXY = shape_thickness / 2
        yK_R = shape_height / 10 * 2
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
        positions = [[0, 0] for i in range(participant_number[x.size])]
        h = 0

    # centerOfMass_shift the shape...
    positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame

    #
    return [my_load.GetWorldPoint(b2Vec2(r)) for r in positions]  # r vectors in the lab frame
