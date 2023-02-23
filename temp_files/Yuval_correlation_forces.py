import matplotlib.pyplot as plt
import numpy as np

from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows
from Classes_Experiment.forces import force_in_frame
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Contact import contact_loop_experiment
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew, Display_loop
from Classes_Experiment.humans import force_from_text
from PhysicsEngine.Contact import find_contact
from Setup.Load import getLoadDim
from Setup.Load import periodicity, centerOfMass_shift, assymetric_h_shift
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy import interpolate
from scipy.stats import pearsonr
import pandas as pd

# x.play()

running, pause = True, False
display = False

# to display single frame
#     Display_renew(screen)
#     Display_loop(load, maze, screen)
#     Display_end()


# forces txt to array
# force_from_text(directory).shape

# to find contact in entire experiment


# if display:
#     screen = Display_screen(maze=maze, caption=x.filename)
#
# i = 0
#
# while i < len(x.frames):
#     x.step(load, i)  # update the position of the load (very simple function, take a look)
#
#     if not pause:
#         contact.append(contact_loop_experiment(load, maze))
#         i += 1
#
#     if display:
#         """Option 1"""
# #         more simplistic, you are just renewing the screen, and displaying the objects
#         Display_renew(screen)
#         Display_loop(load, maze, screen, points=contact[-1])
#
#         """Option 2"""
#         # if you want to be able to pause the display, use this command:
#         # running, i, pause = Pygame_EventManager(x, i, load, maze, screen, pause=pause, points=contact[-1])
#
# if display:
#     Display_end()

global HEIGHT, WIDTH, DISTANCE
HEIGHT, WIDTH, DISTANCE, PROMINENCE = 0, 0, 0, 1

colors = ['#0000FF', '#000000', '#8A2BE2', '#9C661F', '#A52A2A', '#FF4040', '#5F9EA0', '#FF6103', '#ED9121'
    , '#808A87', '#FF7256', '#E50000', '#FFC0CB', '#006400', '#FFD700', '#7FFF00', '#FF7F50', '#13EAC9', '#808000'
    , '#800000', '#E6E6FA', '#D2691E', '#A52A2A', '#C0C0C0', '#029386', '#BBF90F', '#00FFFF', '#6E750E']

ABC_effective_dict = {
    'A': 20, 'B': 21, 'C': 22, 'D': 23, 'E': 24,
    'F': 25, 'G': 0, 'H': 1, 'I': 2, 'J': 3, 'K': 4, 'L': 5, 'M': 6, 'N': 7, 'O': 8, 'P': 9,
    'Q': 10, 'R': 11, 'S': 12, 'T': 13, 'U': 14, 'V': 15, 'W': 16, 'X': 17, 'Y': 18, 'Z': 19
}

medium_sensors_dict = {
    1: 1, 2: 9, 3: 8, 4: 7, 5: 6,
    6: 5, 7: 4, 8: 3, 9: 2
}


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key


def plot_embeddings(x, y, words, marker='x', color='red', label=None):
    """
        Plot in a scatterplot the embeddings of the words specified in the list "words".
        Include a label next to each point.
    """
    for i in range(len(words)):
        x_, y_ = x[i], y[i]
        plt.scatter(x_, y_, marker=marker, color=color, label=label)
        plt.text(x_ + .0003, y_ + .0003, words[i], fontsize=9)


def collision_with_maze_plot(x):
    """
    plot when there is a collision in red and not in blue
    """
    contact = find_contact(x)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    start_to_move, move_fast = cart_started_to_move_frames(x)
    colormap = np.array(['b', 'r'])
    plt.scatter([i for i in range(len(is_frame_in_contact))], np.zeros(len(is_frame_in_contact)),
                c=colormap[is_frame_in_contact])
    plt.scatter([i for i in start_to_move], np.zeros(len(start_to_move)),
                c='g', marker='1')
    plt.scatter([i for i in move_fast], np.zeros(len(move_fast)),
                c='c', marker='1')


def forces_check_func(SOURCE, ADDRESS, action='save', size='M', measured_forces=None, ratio=1):
    """
    you give the location of the file and the address that you want to save the files (only for the TXT of forces,
    the default is for medium
    """
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


def correlation_dataframes(x):
    """
    df2 - the correlations
    df3 - the ab_value of the correlation
    sum_of_corrs_ABSu - mean of the abs values
    """
    raw_forces = [x.participants.frames[k].forces for k in range(len(x.frames))]
    means_of_forces = np.array([np.nan] * len(raw_forces[0]))
    raw_forces_around_mean = np.array([[np.nan] * len(raw_forces[0])] * len(raw_forces))

    for i in range(len(raw_forces[0])):
        force_of_one_sensor = [raw_forces[j][i] for j in range(len(raw_forces))]
        means_of_forces[i] = np.mean(force_of_one_sensor)

    for i in range(len(raw_forces_around_mean)):
        for j in range(len(raw_forces_around_mean[0])):
            result_around_mean = raw_forces[i][j] - means_of_forces[j]
            raw_forces_around_mean[i][j] = result_around_mean

    cross_correlation2 = np.array([[np.nan] * len(raw_forces[0])] * len(raw_forces[0]))
    cross_correlation3 = np.array([[np.nan] * len(raw_forces[0])] * len(raw_forces[0]))
    sum_of_corrs_ABSu = 0

    for i in x.participants.occupied:
        for j in x.participants.occupied:
            force_prod = [raw_forces_around_mean[k][i] * raw_forces_around_mean[k][j] for k in range(len(human_forces))]
            a = [raw_forces[k][i] for k in range(len(raw_forces))]
            b = [raw_forces[k][j] for k in range(len(raw_forces))]
            # force_normalization_factor = gt.np.linalg.norm(a)*gt.np.linalg.norm(b)
            # if force_normalization_factor==0 or np.sum(force_prod)==0:
            #     print("123")
            # correlation_ratio2 = np.divide(np.sum(force_prod), force_normalization_factor)
            correlation_ratio2, _ = pearsonr(a, b)
            cross_correlation2[i][j] = correlation_ratio2
            cross_correlation3[i][j] = np.absolute(correlation_ratio2)
            sum_of_corrs_ABSu += np.absolute(correlation_ratio2)

    mean_of_AB_corrs = sum_of_corrs_ABSu / (len(x.participants.occupied) ** 2)

    if x.size[0] == 'L':
        labels1 = [i + 1 for i in x.participants.occupied]
        labels2 = [((i) % 26) + 1 for i in range(len(raw_forces[0]))]
    elif x.size[0] == 'M':
        labels1 = [medium_sensors_dict[i + 1] for i in x.participants.occupied]
        labels2 = [medium_sensors_dict[i + 1] for i in range(len(raw_forces[0]))]

    df2 = pd.DataFrame(cross_correlation2, index=labels2, columns=labels2)
    df3 = pd.DataFrame(cross_correlation3, index=labels2, columns=labels2)

    return df2, df3, mean_of_AB_corrs


def plot_a_color_matrix_from_df(x, df, min_color=-1, max_color=1):
    f_withOUT_direction = plt.figure(figsize=(8, 6))
    plt.matshow(df, fignum=f_withOUT_direction.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    plt.clim(min_color, max_color)
    cb.ax.tick_params(labelsize=14)
    filename2 = x.filename + '-' + x.participants.condition + '-' + 'withOUT_directions_matrix_pickle'
    plt.title(filename2, fontsize=16)
    plt.show


def single_force_check_func(SOURCE, ADDRESS, sensor, x, action='save', size='M', measured_forces=None, ratio=1):
    sensor_num = x.participants.occupied
    relevant_lines = force_from_text(SOURCE)
    if size == 'L':
        values = [relevant_lines[i][ABC_effective_dict[sensor]] for i in range(len(relevant_lines))]
    elif size == 'M':
        values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
    else:
        print("single_force_check_func: Unknown size")

    index_for_title = sensor_num[sensor] + 1
    # f1 = interp1d(x, y, kind='nearest')
    plt.figure(SOURCE + str(medium_sensors_dict[index_for_title]))
    plt.plot(values, '+')
    plt.title(str(medium_sensors_dict[index_for_title]))
    plt.xlabel('frame number')
    plt.ylabel('Force[]')

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        plt.show()
    else:
        print("action is not valid")

    plt.close


def single_force_check_experiment(ADDRESS, sensor, x, action='save', size='M', filter='NORMAL', measured_forces=None,
                                  ratio=1):
    """ Tabea, this produces the graph with Ofers smooth! """
    sensor_num = x.participants.occupied
    relevant_lines = [x.participants.frames[k].forces for k in range(len(x.frames))]
    frames = x.frames
    if size == 'L':
        values = [relevant_lines[i][ABC_effective_dict[sensor]] for i in range(len(relevant_lines))]
    elif size == 'M':
        values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
    else:
        print("single_force_check_func: Unknown size")

    Frames = np.linspace(frames.min(), frames.max(), 40000)
    X_Y_Spline = interpolate.interp1d(frames, values)
    Values = X_Y_Spline(Frames)
    filtered_values, peaks, bases = ofers_filter(frames, values)
    peaks_val = [values[i] for i in peaks]

    m_val = [(-1 * values[i] + max(values)) for i in range(len(values))]
    peaks2, _ = find_peaks(m_val)
    peaks_val2 = [values[i] for i in peaks2]

    plt.figure(ADDRESS + str(medium_sensors_dict[sensor + 1]))

    if filter == 'NORMAL':
        plt.plot(Frames, filtered_values, color=colors[sensor])
        plt.plot(peaks, peaks_val, 'o', color=colors[sensor])
    elif filter == 'BOX':
        results_half = peak_widths(values, peaks, rel_height=0.8)
        plt.plot(peaks, peaks_val, 'o', color='red')
        plt.hlines(*results_half[1:], color=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
        plt.plot(frames, filtered_values, '+', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
        plt.plot(peaks2, peaks_val2, 'o', c=colors[sensor + 2], label=str(medium_sensors_dict[sensor + 1]))
    elif filter == 'OFERS_FILTER':  # last version
        x_points = np.concatenate((peaks, bases))
        ind = np.argsort(x_points)
        y_vals = [values[i] for i in x_points]
        x_points.sort()
        y_points = [y_vals[i] for i in ind]
        plt.plot(frames, filtered_values, '+', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
        plt.plot(x_points, y_points, '--')
        plt.plot(peaks, peaks_val, 'o', color='red')

    if size == 'L':
        plt.title(sensor + '-' + filter)
    elif size == 'M':
        plt.title(str(medium_sensors_dict[sensor + 1]) + '-' + filter)

    plt.xlabel('frame number')
    plt.ylabel('Force[]')

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        plt.show()
    else:
        print("action is not valid")

    plt.close()


def all_forces_toghether_exp(SOURCE, ADDRESS, x, action='save', size='M', filter='NORMAL', measured_forces=None,
                             ratio=1):
    """ Tabea, this produces the graph with Ofers smooth! """
    sensor_num = x.participants.occupied
    relevant_lines = [x.participants.frames[k].forces for k in range(len(x.frames))]
    frames = x.frames

    fig, ax = plt.subplots()

    for sensor in range(len(relevant_lines[0])):

        if size == 'L':
            values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
        elif size == 'M':
            values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
            if (min(values) < 0):
                factor = abs(min(values))
                values = [(values[i] + factor) for i in range(len(values))]
        else:
            print("single_force_check_func: Unknown size")

        Frames = np.linspace(frames.min(), frames.max(), 40000)
        X_Y_Spline = interpolate.interp1d(frames, values)
        Values = X_Y_Spline(Frames)

        filtered_values, peaks, bases = ofers_filter(frames, values, def_gap=30)
        peaks_val = [values[i] for i in peaks]

        # plt.plot(peaks, peaks_val, 'o', c=colors[sensor], label='peaks of ' + str(medium_sensors_dict[sensor+1]))

        # if medium_sensors_dict[sensor+1]==7:
        #     ax.plot(Frames, Values,'+', c=colors[sensor],  label=str(medium_sensors_dict[sensor+1]))

        if (filter == 'NORMAL'):
            if size == 'M':
                plot_embeddings(peaks, peaks_val, str(medium_sensors_dict[sensor + 1]) * len(peaks), marker='o',
                                color=colors[sensor],
                                label='peaks of ' + str(medium_sensors_dict[sensor + 1]))
                ax.plot(Frames, Values, '+', c=colors[sensor],
                        label=str(medium_sensors_dict[sensor + 1]))  # with the linspace
                # ax.plot(frames, values, '+', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1])) #without the linspace
            elif size == 'L':
                plot_embeddings(peaks, peaks_val, get_key(ABC_effective_dict, sensor + 1) * len(peaks), marker='o',
                                color=colors[sensor],
                                label='peaks of ' + str(get_key(ABC_effective_dict, sensor + 1)))
                ax.plot(Frames, Values, '+', c=colors[sensor],
                        label=str(get_key(ABC_effective_dict, sensor + 1)))  # with the linspace
            ax.set_xlabel('frame number')
            ax.set_ylabel('Force[]')
            ax.set_title('NORMAL')

        elif (filter == 'BOX'):
            # plot_embeddings(peaks, peaks_val,  str(medium_sensors_dict[sensor+1])*len(peaks), marker='o', color=colors[sensor],
            #                 label='peaks of ' + str(medium_sensors_dict[sensor+1]))
            results_half = peak_widths(values, peaks, rel_height=0.4)
            if size == 'M':
                plt.hlines(*results_half[1:], color=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
                # ax.plot(Frames, Values, '+', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
            elif size == 'L':
                plt.hlines(*results_half[1:], color=colors[sensor], label=str(get_key(ABC_effective_dict, sensor + 1)))
            ax.set_xlabel('frame number')
            ax.set_ylabel('Force[]')
            leg = ax.legend()
            ax.set_title('BOX')

        elif (filter == 'OFERS_FILTER'):
            x_points = np.concatenate((peaks, bases))
            ind = np.argsort(x_points)
            y_vals = [values[i] for i in x_points]
            x_points.sort()
            y_points = [y_vals[i] for i in ind]
            # ax.plot(frames, filtered_values, '+', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
            if size == 'M':
                ax.plot(x_points, y_points, '--', c=colors[sensor], label=str(medium_sensors_dict[sensor + 1]))
            elif size == 'L':
                ax.plot(x_points, y_points, '--', c=colors[sensor], label=str(get_key(ABC_effective_dict, sensor + 1)))
            ax.set_xlabel('frame number')
            ax.set_ylabel('Force[]')
            leg = ax.legend()

    collision_with_maze_plot(x=x)

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        plt.show()
    else:
        print("action is not valid")

    plt.close


def ofers_filter(x_array, y_array, def_gap=15):
    """
    signal below 5% of maximum excluded
    tries to find the peaks (if two or more are too close it unifies them
    """
    threshold = (max(y_array) - min(y_array)) * 0.05 + min(y_array)
    filtered = [None] * len(y_array)
    for i in range(len(x_array)):
        if y_array[i] > threshold:
            filtered[i] = y_array[i]
        else:
            filtered[i] = np.nan

    X_array = np.linspace(x_array.min(), x_array.max(), 40000)
    X_Y_Spline = interpolate.interp1d(x_array, filtered)
    Y_array = X_Y_Spline(X_array)

    peaks, _ = find_peaks(filtered, height=0.6, width=WIDTH)
    mid_calc = [peaks[i] for i in range(len(peaks)) if peaks[i] > threshold]
    peaks_val = [y_array[i] for i in peaks]
    prominences, left_bases, right_bases = peak_prominences(filtered, peaks, wlen=300)

    bases = list()

    if (len(left_bases) and len(right_bases)):
        bases.append(left_bases[0])

        for i in range(len(left_bases) - 1):
            if i <= len(left_bases) - 2:
                if left_bases[i] == left_bases[i + 1]:
                    continue
            if (left_bases[i + 1] - right_bases[i]) > def_gap:
                bases.append(right_bases[i])
                bases.append(left_bases[i + 1])

        bases.append(right_bases[-1])

    return filtered, peaks, bases


def all_forces_toghether_check(SOURCE, ADDRESS, x, action='save', size='M', measured_forces=None, ratio=1):
    sensor_num = x.participants.occupied
    relevant_lines = force_from_text(SOURCE)

    fig, ax = plt.subplots()

    for sensor in range(len(relevant_lines[0])):

        if size == 'L':
            values = [relevant_lines[i][ABC_effective_dict[sensor]] for i in range(len(relevant_lines))]
        elif size == 'M':
            values = [relevant_lines[i][sensor] for i in range(len(relevant_lines))]
        else:
            print("single_force_check_func: Unknown size")

        peaks, _ = find_peaks(values, height=4, width=10)
        peaks_val = [values[i] for i in peaks]
        plt.plot(peaks, peaks_val, 'o', label='peaks of ' + str(medium_sensors_dict[sensor + 1]))
        ax.plot(values, '+', label=str(medium_sensors_dict[sensor + 1]))
        ax.set_xlabel('frame number')
        ax.set_xlabel('Force[]')

    if action == 'save':
        plt.savefig(ADDRESS)
    elif action == 'open':
        leg = ax.legend();
        plt.show()
    else:
        print("action is not valid")

    plt.close()


def theta_trajectory(twoD_vec, unit='RADIAN'):
    # TODO: Yuval: to go over all the functions that use this one (I changed the output to degrees)
    '''
    :param
    :return: the angle of the vector (in degrees)
    '''
    angle = np.arctan2(twoD_vec[1], twoD_vec[0])
    if angle < 0:
        angle += (2 * np.pi)
    degrees = angle * 180 / np.pi
    if unit == 'RADIAN':
        return angle
    elif unit == 'DEGREE':
        return degrees


def calc_angle_diff(alpha, beta):
    """
    Calculates the angular difference between two angles calc_alpha and calc_beta (floats, in degrees).
    returns d_alpha: (float, in degrees between 0 and 360)
    """
    d = alpha - beta
    d_alpha = (d + 180) % 360 - 180
    return d_alpha


def numeral_velocity(obj, i, step=None):
    # the step must be an even number
    if (step == None):
        step = obj.fps

    return obj.position[min(i + int(np.divide(step, 2)), obj.position.shape[0] - 1)] - \
           obj.position[min(i - int(np.divide(step, 2)), obj.position.shape[0] - 1)]


def cart_started_to_move_frames(obj):
    """
    maybe you can play with the parameters of the lines of the variables - not_moving, move_fast
    """
    velocity = [numeral_velocity(obj, i) for i in range(len(obj.frames))]
    power = [np.linalg.norm(velocity[i]) for i in range(len(velocity))]
    not_moving = [i for i in range(len(velocity)) if power[i] < 0.015]
    move_fast = [i for i in range(len(velocity)) if power[i] > 0.28]
    start_movement = [(not_moving[i] + 1) for i in range(len(not_moving) - 1) if
                      (not_moving[i + 1] - not_moving[i]) > 30]

    return start_movement, move_fast


def numeral_angular_velocity(obj, i):
    """
    work like crappy velocity but with angles
    """
    vel = calc_angle_diff(obj.angle[min(i + int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)], \
                          obj.angle[min(i - int(np.divide(obj.fps, 2)), obj.position.shape[0] - 1)])
    res = vel % (2 * (np.pi))
    if res > 6:
        res = 2 * (np.pi) - res
    return res


def normalized_dot_prod(vec_A, vec_B):
    dot_prod = (vec_A[0] * vec_B[0] + vec_A[1] * vec_B[1])
    abs_val = np.linalg.norm(vec_A) * np.linalg.norm(vec_B)
    return dot_prod / abs_val


def dot_prod(vec_A, vec_B):
    dot_prod = (vec_A[0] * vec_B[0] + vec_A[1] * vec_B[1])
    return dot_prod


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
    """
    similar to force_vector_positions which Tabea wrote
    maybe it's better to merge it with your function
    """
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
        h = centerOfMass_shift * shape_width

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
        h = centerOfMass_shift * shape_width

    else:
        positions = [[0, 0] for i in range(participant_number[x.size])]
        h = 0

    # shift the shape...
    positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame

    return positions


def first_method_graphs(x):
    """ first sanity check """
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('forces and cart movement test - FIRST METHOD', fontsize=16)

    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]
    human_total_force = (np.sum(human_forces, axis=1)).reshape(len(human_forces), 2)
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


def second_method_graphs(x, force_treshhold=0.5):
    """ second sanity check """
    contact = find_contact(x, display=False)
    is_frame_in_contact = [int(len(contact[i]) != 0) for i in range(len(contact))]
    colormap = np.array(['b', 'r'])

    fig, axs = plt.subplots(constrained_layout=True)
    # fig.suptitle('forces and cart movement test - SECOND METHOD', fontsize=16)

    cart_velocity = [numeral_velocity(x, i) for i in range(len(x.frames))]
    human_forces = [force_in_frame(x, i) for i in range(len(x.frames))]

    human_total_force = (np.sum(human_forces, axis=1)).reshape(9097, 2)
    human_total_force_treshhold = [human_total_force[i] * (np.linalg.norm(human_total_force[i]) > force_treshhold)
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
