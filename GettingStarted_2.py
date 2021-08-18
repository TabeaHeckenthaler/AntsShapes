import matplotlib.pyplot as plt
import numpy as np

from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows
from Classes_Experiment.forces import force_in_frame
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Contact import Contact_loop
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop
from Classes_Experiment.humans import force_from_text
from PhysicsEngine.Contact import find_contact


''' Display a experiment '''
# names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
solver = 'human'
x = Get('medium_20201221111935_20201221112858', solver)
x.participants = Humans(x)
# x.play(forces=[participants_force_arrows])
# press Esc to stop the display

''' Find contact points '''
contact = []
my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver)
my_load = Load(my_maze, position=x.position[0])

# x.play()

running, pause = True, False
display = False

# to display single frame
#     Display_renew(screen)
#     Display_loop(my_load, my_maze, screen)
#     Display_end()


# forces txt to array
# force_from_text(directory).shape

# to find contact in entire experiment


# if display:
#     screen = Display_screen(my_maze=my_maze, caption=x.filename)
#
# i = 0
#
# while i < len(x.frames):
#     x.step(my_load, i)  # update the position of the load (very simple function, take a look)
#
#     if not pause:
#         contact.append(Contact_loop(my_load, my_maze))
#         i += 1
#
#     if display:
#         """Option 1"""
# #         more simplistic, you are just renewing the screen, and displaying the objects
#         Display_renew(screen)
#         Display_loop(my_load, my_maze, screen, points=contact[-1])
#
#         """Option 2"""
#         # if you want to be able to pause the display, use this command:
#         # running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=pause, points=contact[-1])
#
# if display:
#     Display_end()

def forces_check_func(SOURCE, ADDRESS):
    relevant_lines = force_from_text(SOURCE)
    max_index, max_values = [np.argmax(relevant_lines[i]) for i in range(len(relevant_lines))], \
                            [max(relevant_lines[i]) for i in range(len(relevant_lines))]
    values_ordered = np.dstack((max_index, max_values))
    ABC = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  #gives an array of the ABC...
    num_of_relevat_graphes = len(set(max_index))

    measurements = [-1, 3.9, 7.4, 11.3, 11.3, -1, 16.5, 6.3, 2.1, 5.3, 9.1, 11.1, 4.3, 5.6, 2.9, 4.3, 2.2,
                    1, -2, 10, 6.3, 6, 12.1, 5.4, 6.5, 6.5] # -1 means broken, -2 means a short pulse

    fig, axes = plt.subplots((round(np.sqrt(num_of_relevat_graphes)) + 1),
                             round(np.sqrt(num_of_relevat_graphes)), figsize=(9, 9))  # Create a figure with all the relevant data
    axes = axes.flatten()  # Convert the object storing the axes for a 3 by 3 array into a 1D array.
    fig.tight_layout(pad=5.0) # padding between the subplots
    counter = 0  #counter for the loop that generate the graphes

    for i in range(1, 27):
        data = []

        for j in range(len(max_index)):
            if values_ordered[0][j][0] == i:
                data.append(values_ordered[0][j][1])
        if (data):
            axes[counter].plot(data,'+')
            axes[counter].plot((measurements[(i+7)%len(measurements)]*(np.ones(50))), 'g')
            axes[counter].set_title(ABC[(i+7)%len(ABC)]) #beacause it's start with 'H'
            axes[counter].set_xlabel('frame number')
            axes[counter].set_ylabel('Force[]')
            counter +=1

    plt.savefig(ADDRESS)


forces_check_func('force_check\\150521.TXT', 'force_check\\force_detector_check_trash.png')
forces_check_func('force_check\\150608.TXT', 'force_check\\force_detector_check_W.png')
forces_check_func('force_check\\150624.TXT', 'force_check\\force_detector_check_WXY.png')
forces_check_func('calibration_exp.TXT', 'force_detector_check5.png')


def theta_trajectory(twoD_vec):
    '''

    :param
    :return: the angle of the vector (in radians)
    '''

    x_comp, y_comp = zip(*twoD_vec)
    return np.degrees(np.arctan2(x_comp, y_comp))


def calc_angle_diff(alpha, beta):
    """
    Calculates the angular difference between two angles alpha and beta (floats, in degrees).
    returns d_alpha: (float, in degrees between 0 and 360)
    """
    d = alpha - beta
    d_alpha = (d + 180) % 360 - 180
    return d_alpha


def numeral_velocity(obj, i):
    return obj.position[min(i + int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)] - \
           obj.position[min(i - int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)]


def abs_value_2D(twoD_vec):
    return np.sqrt(twoD_vec[0]**2 + twoD_vec[1]**2)


def normalized_dot_prod(vec_A, vec_B):
    dot_prod = (vec_A[0]*vec_B[0] + vec_A[1]*vec_B[1])
    abs_val = abs_value_2D(vec_A)*abs_value_2D(vec_B)
    return dot_prod/abs_val


def first_method_graphes():
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('forces and cart movement test - FIRST METHOD', fontsize=16)

    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]
    human_total_force = (np.sum(human_forces, axis=1)).reshape(9097,2)
    theta_human = np.unwrap(theta_trajectory(human_total_force))
    axs[0].plot(theta_human, 'b')
    axs[0].scatter([i for i in range(len(is_frame_in_contact))],np.zeros(len(is_frame_in_contact)) , c = colormap[is_frame_in_contact])
    axs[0].set_title('total human direction')
    axs[0].set_xlabel('frame number')
    axs[0].set_ylabel('direction [degrees]')

    cart_velocity = [numeral_velocity(x, i) for i in range(len(x.frames))]
    theta_cart = np.unwrap(theta_trajectory(cart_velocity))

    axs[1].plot(theta_cart, 'g')
    axs[1].scatter([i for i in range(len(is_frame_in_contact))],np.zeros(len(is_frame_in_contact)) , c = colormap[is_frame_in_contact])
    axs[1].set_title('cart direction')
    axs[1].set_xlabel('frame number')
    axs[1].set_ylabel('direction [degrees]')

    velocity_compare = calc_angle_diff(theta_human, theta_cart)

    axs[2].plot(velocity_compare, 'r')
    axs[2].scatter([i for i in range(len(is_frame_in_contact))],np.zeros(len(is_frame_in_contact)) , c = colormap[is_frame_in_contact])
    axs[2].set_title('subtraction between cart and human power')
    axs[2].set_xlabel('frame number')
    axs[2].set_ylabel('direction [degrees]')
    # plt.savefig('fixed_arrows.png')
    plt.show()


def second_method_graphes(force_treshhold = 0.5):
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(constrained_layout=True)
    # fig.suptitle('forces and cart movement test - SECOND METHOD', fontsize=16)

    cart_velocity = [numeral_velocity(x, i) for i in range(len(x.frames))]
    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]

    human_total_force = (np.sum(human_forces, axis=1)).reshape(9097,2)
    human_total_force_treshhold = [human_total_force[i]*(abs_value_2D(human_total_force[i]) > force_trashhold)
                                   for i in range(len(human_total_force))]

    dot_prodacts = [normalized_dot_prod(cart_velocity[i],human_total_force_trashhold[i]) for i in range(len(human_forces))]

    axs.plot(dot_prodacts, 'b')
    axs.scatter([i for i in range(len(is_frame_in_contact))],np.zeros(len(is_frame_in_contact)) , c = colormap[is_frame_in_contact])
    axs.set_title('forces and cart movement test - SECOND METHOD')
    axs.set_xlabel('frame number')
    axs.set_ylabel('direction [degrees]')

    # plt.savefig('fixed_arrows.png')
    plt.show()
# press Esc to stop the display
second_method_graphes(force_treshhold = 2)