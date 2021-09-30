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
from Setup.Load import getLoadDim
from Setup.Load import periodicity, shift, assymetric_h_shift

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

def forces_check_func(SOURCE, ADDRESS, action='save', size='M', measured_forces=None, ratio=1):
    relevant_lines = force_from_text(SOURCE)
    max_index, max_values = [np.argmax(relevant_lines[i]) for i in range(len(relevant_lines))], \
                            [max(relevant_lines[i]) for i in range(len(relevant_lines))]
    values_ordered = np.dstack((max_index, max_values))
    ABC = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # gives an array of the ABC...
    num_of_relevat_graphes = len(set(max_index))

    fig, axes = plt.subplots((round(np.sqrt(num_of_relevat_graphes)) + 1),
                             round(np.sqrt(num_of_relevat_graphes)),
                             figsize=(9, 9))  # Create a figure with all the relevant data
    axes = axes.flatten()  # Convert the object storing the axes for a 3 by 3 array into a 1D array.
    fig.tight_layout(pad=5.0)  # padding between the subplots
    counter = 0  # counter for the loop that generate the graphes

    for i in range(len(ABC)):
        data = list()

        for j in range(len(max_index)):
            if values_ordered[0][j][0] == i:
                data.append(values_ordered[0][j][1])
        if (data):
            axes[counter].plot(data, '+')
            if measured_forces:
                axes[counter].plot((measured_forces[(i + 6) % len(measured_forces)] * (np.ones(50)) * ratio), 'g')
            axes[counter].set_title(ABC[(i + 6) % len(ABC)])  # beacause it's start with 'H'
            axes[counter].set_xlabel('frame number')
            axes[counter].set_ylabel('Force[]')
            counter += 1

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        plt.show()
    else:
        print("action is not valid")


def single_force_check_func(SOURCE, ADDRESS, sensor, action='save', size='M', measured_forces=None, ratio=1):
    ABC_effective_dict = {
        'A': 20, 'B': 21, 'C': 22, 'D': 23, 'E': 24,
        'F': 25, 'G': 0, 'H': 1, 'I': 2, 'J': 3, 'K': 4, 'L': 5, 'M': 6, 'N': 7, 'O': 8, 'P': 9,
        'Q': 10, 'R': 11, 'S': 12, 'T': 13, 'U': 14, 'V': 15, 'W': 16, 'X': 17, 'Y': 18, 'Z': 19
    }

    relevant_lines = force_from_text(SOURCE)
    if size == 'L':
        values = [relevant_lines[i][ABC_effective_dict[sensor]] for i in range(len(relevant_lines))]
    elif size == 'M':
        values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
    else:
        print("single_force_check_func: Unknown size")

    plt.figure(SOURCE + str(sensor))
    plt.plot(values)
    plt.title(str(sensor))
    plt.xlabel('frame number')
    plt.ylabel('Force[]')

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        plt.show()
    else:
        print("action is not valid")

    plt.close


def theta_trajectory(twoD_vec):
    '''

    :param
    :return: the angle of the vector (in radians)
    '''

    x_comp, y_comp = zip(*twoD_vec)
    return np.degrees(np.arctan2(y_comp, x_comp))


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


def numeral_angular_velocity(obj, i):
    vel = calc_angle_diff(obj.angle[min(i + int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)], \
                          obj.angle[min(i - int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)])
    res = vel % (2 * (np.pi))
    if res > 6:
        res = 2 * (np.pi) - res
    return res


def abs_value_2D(twoD_vec):
    return np.sqrt(twoD_vec[0] ** 2 + twoD_vec[1] ** 2)


def normalized_dot_prod(vec_A, vec_B):
    dot_prod = (vec_A[0] * vec_B[0] + vec_A[1] * vec_B[1])
    abs_val = abs_value_2D(vec_A) * abs_value_2D(vec_B)
    return dot_prod / abs_val


def cross_prod(vec_A, vec_B):
    """
    It's important to notice that the action is (A X B) and NOT!!!! ---> (B X A)
    and because in our case the maze is only 2D, the prod will always be in Z direction
    """
    prod = vec_A[0] * vec_B[1] - vec_A[1] * vec_B[0]
    return prod


def sum_of_cross_prods(vec_A, vec_B):
    if len(vec_A) != len(vec_B):
        print("sum_of_cross_prods: Not in the same length")
        return

    prod = [cross_prod(vec_A[i], vec_B[i]) for i in range(len(vec_A))]
    return np.sum(prod)


def vector_rotation(vec_A, radian_angle):
    if len(vec_A) != 2:
        vec_A = np.transpose(vec_A)

    if len(vec_A) != 2:
        print("There is a dimension problem")
        return

    rotation_matrix = [[np.cos(radian_angle), -np.sin(radian_angle)], [np.sin(radian_angle), np.cos(radian_angle)]]
    return np.matmul(rotation_matrix, vec_A)


def force_vector_positions_In_LOAD_FRAME(my_load, x):
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

    return positions


def torque_in_load(my_load, x, force_vector_In_Lab_Frame, amgle_In_rads):
    r_positions = [(force_vector_positions_In_LOAD_FRAME(my_load, x))[name] for name in x.participants.occupied]
    forces_in_load_frame = [vector_rotation(force_vector_In_Lab_Frame[i], amgle_In_rads) for i in
                            range(len(force_vector_In_Lab_Frame))]
    return sum_of_cross_prods(r_positions, forces_in_load_frame)


def first_method_graphes(x):
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('forces and cart movement test - FIRST METHOD', fontsize=16)

    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]
    human_total_force = (np.sum(human_forces, axis=1)).reshape(9097, 2)
    theta_human = np.unwrap(theta_trajectory(human_total_force))
    axs[0].plot(theta_human, 'b')
    axs[0].scatter([i for i in range(len(is_frame_in_contact))], np.zeros(len(is_frame_in_contact)),
                   c=colormap[is_frame_in_contact])
    axs[0].set_title('total human direction')
    axs[0].set_xlabel('frame number')
    axs[0].set_ylabel('direction [degrees]')

    cart_velocity = [numeral_velocity(x, i) for i in range(len(x.frames))]
    theta_cart = np.unwrap(theta_trajectory(cart_velocity))

    axs[1].plot(theta_cart, 'g')
    axs[1].scatter([i for i in range(len(is_frame_in_contact))], np.zeros(len(is_frame_in_contact)),
                   c=colormap[is_frame_in_contact])
    axs[1].set_title('cart direction')
    axs[1].set_xlabel('frame number')
    axs[1].set_ylabel('direction [degrees]')

    velocity_compare = calc_angle_diff(theta_human, theta_cart)

    axs[2].plot(velocity_compare, 'r')
    axs[2].scatter([i for i in range(len(is_frame_in_contact))], np.zeros(len(is_frame_in_contact)),
                   c=colormap[is_frame_in_contact])
    axs[2].set_title('subtraction between cart and human power')
    axs[2].set_xlabel('frame number')
    axs[2].set_ylabel('direction [degrees]')
    # plt.savefig('fixed_arrows.png')
    plt.show()


def second_method_graphes(x, force_treshhold=0.5):
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(constrained_layout=True)
    # fig.suptitle('forces and cart movement test - SECOND METHOD', fontsize=16)

    cart_velocity = [numeral_velocity(x, i) for i in range(len(x.frames))]
    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]

    human_total_force = (np.sum(human_forces, axis=1)).reshape(9097, 2)
    human_total_force_treshhold = [human_total_force[i] * (abs_value_2D(human_total_force[i]) > force_treshhold)
                                   for i in range(len(human_total_force))]

    dot_prodacts = [normalized_dot_prod(cart_velocity[i], human_total_force_treshhold[i]) for i in
                    range(len(human_forces))]

    axs.plot(dot_prodacts, 'b')
    axs.scatter([i for i in range(len(is_frame_in_contact))], np.zeros(len(is_frame_in_contact)),
                c=colormap[is_frame_in_contact])
    axs.set_title('forces and cart movement test - SECOND METHOD')
    axs.set_xlabel('frame number')
    axs.set_ylabel('direction [degrees]')

    # plt.savefig('fixed_arrows.png')
    plt.show()
# press Esc to stop the display
