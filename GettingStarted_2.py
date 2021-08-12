from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Contact import Contact_loop
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop

''' Display a experiment '''
# names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
solver = 'human'
x = Get('medium_20201221135753_20201221140218', solver)
# x.participants = Humans(x)
# x.play(forces=[participants_force_arrows])
# press Esc to stop the display

''' Find contact points '''
contact = []
my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
my_load = Load(my_maze, position=x.position[0])
screen = Display_screen(my_maze=my_maze)
running, pause = True, False
display = False

# to display single frame
Display_renew(screen)
Display_loop(my_load, my_maze, screen)
Display_end()

# to find contact in entire experiment
if display:
    screen = Display_screen(my_maze=my_maze)

i = 0
while i < len(x.frames):
    x.step(my_load, i)  # update the position of the load (very simple function, take a look)

    if not pause:
        contact.append(Contact_loop(my_load, my_maze))
        i += 1

    if display:
        """Option 1"""
        # more simplistic, you are just renewing the screen, and displaying the objects
        Display_renew(screen)
        Display_loop(my_load, my_maze, screen, points=contact[-1])

        """Option 2"""
        # if you want to be able to pause the display, use this command:
        # running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=pause, points=contact[-1])

if display:
    Display_end()


