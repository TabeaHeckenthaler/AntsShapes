from Setup.Maze import Maze
from Setup.Load import Load, Loops
import numpy as np
from matplotlib import pyplot as plt
from PhysicsEngine.Contact import find_contact
from trajectory_inheritance.trajectory import get

solver = 'human'
x = get('large_20210419100024_20210419100547', solver)
# x.play()

''' find vertices '''
my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
my_load = Load(my_maze, position=x.position[0])

# my_maze contains the maze and the load!
maze_dict = {body.userData: i for i, body in enumerate(my_maze.bodies)}
# vertices will contain lists of lists with 4 points. These are the corners of the rectangles!
maze_vertices = np.array(Loops(my_maze.bodies[maze_dict['my_maze']]))
load_vertices = np.array(Loops(my_maze.bodies[maze_dict['my_load']]))

''' plot vertices '''
plt.scatter(maze_vertices.reshape(4*len(maze_vertices), 2)[:, 0],
            maze_vertices.reshape(4*len(maze_vertices), 2)[:, 1])
plt.scatter(load_vertices.reshape(4*len(load_vertices), 2)[:, 0],
            load_vertices.reshape(4*len(load_vertices), 2)[:, 1])


''' find contact points '''
contact = find_contact(x, display=False)

# [i for i, l in enumerate(contact) if len(l)>0]
# '''to display single frame'''
# screen = Display_screen(my_maze=my_maze, caption=x.filename)
# Display_renew(screen)
# Display_loop(my_load, my_maze, screen)
# Display_end()

x.play()