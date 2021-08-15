from trajectory import Get
from Setup.Maze import Maze
from Setup.Load import Load, Loops
from PhysicsEngine.Display_Pygame import Display_screen, Display_end, Display_renew, Display_loop
import numpy as np
from matplotlib import pyplot as plt
from PhysicsEngine.Contact import find_contact

solver = 'human'
x = Get('medium_20201221135753_20201221140218', solver)
x.play()

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

'''to display single frame'''
screen = Display_screen(my_maze=my_maze)
Display_renew(screen)
Display_loop(my_load, my_maze, screen)
x.play( )
Display_end()
