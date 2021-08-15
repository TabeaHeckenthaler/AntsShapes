from trajectory import Get
from Box2D import b2Vec2
from Setup.Load import Load, Loops
from Setup.Maze import Maze
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from Setup.MazeFunctions import BoxIt

# # Get a specific experiment
x = Get('S_SPT_4350023_SSpecialT_1_ants', 'ant')
#
# # Display the experiment
# x, contact = x.play(1, 'Display', 'contact')
#
# # most important attributes of x are accessed by....
# position = x.position # center of mass in x and y for every frame
# angle = x.angle # angle for every frame
# filename = x.filename # differs from original filename only by underscores between the size and the shape
# VideoChain = x.VideoChain # relevant if we have multiple parts...
#
# # How to get the ant information
# ants = Ants(x.filename, x.old_filenames(0))
# ants = ants.matlab_ant_loading(x)
#
# # only if I have multiple video parts
# for old_name in [x.old_filenames(i) for i in range(1,len(x.VideoChain))]:
#     ants2 = Ants(x.filename, old_name)
#     ants = ants + ants2.matlab_ant_loading(x)
#
# # to access data of ants of frame 5 for example, ...
# ant_positions = ants.frames[5].position
# ant_angle = ants.frames[5].angle # in radian
# ant_carrying = ants.frames[5].carrying
# ant_major_axis_length = ants.frames[5].major_axis_length
# pix2cm = ants.pix2cm

# edge of contact
my_load = Load(Maze(size=x.size, shape=x.shape, solver=x.solver))
my_load.position = b2Vec2(0, 0)
my_load.angle = 0
load_vertices, lines = Loops(my_load)

fig = plt.figure()

edge_points = []
for NumFixture in range(int(len(load_vertices) / 4)):
    # add a debug point to 'debug through this loop', to see, what the order of points is.
    # You still will have to do some work here to sort them properly...
    edge_points = edge_points + BoxIt(load_vertices[NumFixture * 4:(NumFixture + 1) * 4], 0.001).tolist()
    load_tree = cKDTree(edge_points)
    plt.plot(load_tree.data[:, 0], load_tree.data[:, 1], '.')
