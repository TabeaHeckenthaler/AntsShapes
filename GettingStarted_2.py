from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Contact import Contact_loop
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop
from PhysicsEngine.Contact import find_contact
from trajectory import Get
from Classes_Experiment.forces import force_in_frame
from Analysis_Functions.Velocity import crappy_velocity
import matplotlib.pyplot as plt
import numpy as np
''' Display a experiment '''
# names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
solver = 'human'
x = Get('medium_20201221135753_20201221140218', solver)
x.participants = Humans(x)
#x.play(forces=[participants_force_arrows])
# press Esc to stop the display

''' Find contact points '''
contact = find_contact(x)
print(np.where(contact))


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

