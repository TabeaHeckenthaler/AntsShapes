import matplotlib.pyplot as plt
import numpy as np

from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Contact import Contact_loop
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop
from Classes_Experiment.humans import force_from_text
import re


''' Display a experiment '''
# names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
solver = 'human'
x = Get('medium_20201221111935_20201221112858', solver)
x.participants = Humans(x)
x.play(forces=[participants_force_arrows])
# press Esc to stop the display

''' Find contact points '''
contact = []
my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
my_load = Load(my_maze, position=x.position[0])
screen = Display_screen(my_maze=my_maze, caption=x.filename)
running, pause = True, False
display = True

# to display single frame
Display_renew(screen)
Display_loop(my_load, my_maze, screen)
Display_end()


# forces txt to array
# force_from_text(directory).shape

# to find contact in entire experiment

def find_entire_contacts(x, load):
    if display:
        screen = Display_screen(my_maze=my_maze, caption=x.filename)

    i = 0
    while i < len(x.frames):
        x.step(load, i)  # update the position of the load (very simple function, take a look)

        if not pause:
            contact.append(Contact_loop(load, my_maze))
            i += 1

        if display:
            """Option 1"""
            # more simplistic, you are just renewing the screen, and displaying the objects
            # Display_renew(screen)
            # Display_loop(load, my_maze, screen, points=contact[-1])

            """Option 2"""
            # if you want to be able to pause the display, use this command:
            running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=pause, points=contact[-1])

    if display:
        Display_end()


relevant_lines = force_from_text('calibration_exp.TXT')
max_index, max_values = [np.argmax(relevant_lines[i]) for i in range(len(relevant_lines))], \
                        [max(relevant_lines[i]) for i in range(len(relevant_lines))]
values_ordered = np.dstack((max_index, max_values))
ABC = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

num_of_relevat_graphes = len(set(max_index))

fig, axes = plt.subplots((round(np.sqrt(num_of_relevat_graphes)) + 1), round(np.sqrt(num_of_relevat_graphes)), figsize=(9, 9))  # Create a figure with 3 over 3 axes
axes = axes.flatten()  # Convert the object storing the axes for a 3 by 3 array into a 1D array of lenth 9.
fig.tight_layout(pad=5.0)
counter = 0

for i in range(1, 27):
    data = []

    for j in range(len(max_index)):
        if values_ordered[0][j][0] == i:
            data.append(values_ordered[0][j][1])
    if (data):
        axes[counter].plot(data,'+')
        axes[counter].set_title(ABC[(i+11)%len(ABC)]) #beacause it's start with 'L'
        axes[counter].set_xlabel('frame number')
        axes[counter].set_ylabel('Force[]')
        counter +=1

plt.savefig('force_detector_check.png')

#
# for i_field, field in enumerate(x_fields):
#     ax = axes[i_field]
#     ax.plot(dataset[field].values, dataset['Rented Bike Count'].values, '.', markersize=1)
#     ax.set_xlabel(field)
#     ax.set_ylabel('Rented Bike Count')
#
# def theta_trajectory(twoD_vec):
#     '''
#
#     :param
#     :return: the angle of the vector (in radians)
#     '''
#
#     x_comp, y_comp = zip(*twoD_vec)
#     return np.degrees(np.arctan2(x_comp, y_comp))
#
#
# def calc_angle_diff(alpha, beta):
#     """
#     Calculates the angular difference between two angles alpha and beta (floats, in degrees).
#     returns d_alpha: (float, in degrees between 0 and 360)
#     """
#     d = alpha - beta
#     d_alpha = (d + 180) % 360 - 180
#     return d_alpha
#
#
def numeral_velocity(obj, i):
    return obj.position[min(i + int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)] - \
           obj.position[min(i - int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)]

#
# fig, axs = plt.subplots(3, 1, constrained_layout=True)
# fig.suptitle('forces and cart movement test', fontsize=16)
#
# human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]
# human_total_force = np.sum(human_forces, axis=1)
# theta_human = np.unwrap(theta_trajectory(human_total_force))
# axs[0].plot(theta_human, 'b')
# axs[0].set_title('total human direction')
# axs[0].set_xlabel('frame number')
# axs[0].set_ylabel('direction [degrees]')
#
# cart_velocity = [crappy_velocity(x, i) for i in range(len(x.frames))]
# theta_cart = np.unwrap(theta_trajectory(cart_velocity))
#
# axs[1].plot(theta_cart, 'g')
# axs[1].set_title('cart direction')
# axs[1].set_xlabel('frame number')
# axs[1].set_ylabel('direction [degrees]')
#
# velocity_compare = calc_angle_diff(theta_human, theta_cart)
#
# axs[2].plot(velocity_compare, 'r')
# axs[2].set_title('subtraction between cart and human power')
# axs[2].set_xlabel('frame number')
# axs[2].set_ylabel('direction [degrees]')
# plt.savefig('unwrap.png')


# press Esc to stop the display
