import numpy as np
from Setup.MazeFunctions import ConnectAngle
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from Analysis_Functions.GeneralFunctions import ranges, gauss
from matplotlib import pyplot as plt

max_Vel_trans, max_Vel_angle = {'XS': 4, 'S': 4, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}, \
                               {'XS': 10, 'S': 10, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}


# StartedScripts: Are all the trajectories in the right direction??
# StartedScripts: rewrite the velocity module... (SH_4170007_SmallHT_2_ants, SH_4170006_smallHT_2_ants.mat for example)

def crappy_velocity(x, i):
    return x.position[min(i + 30, x.position.shape[0] - 1)] - x.position[i]


def velocity_arrow(x, my_load, i):
    start = x.position[i]
    end = x.position[i] + 10 * crappy_velocity(x, i)
    string = 'v'
    return [(start, end, string)]


def velocity(position, angle, fps, size, shape, second_smooth, solver, *args, **kwargs):
    from Setup.Load import average_radius
    """ If I add specifications to args, I only return these specific velocities,
    otherwise, I return x, y and angular velocity"""

    ''' Returns velocity vectors x and y in units cm/s '''
    smoother = second_smooth * (fps - np.mod(fps, 2) + 1)  # smoother has to be an odd number

    locations = np.zeros(position.shape[0])
    units = []

    # we have to use a function called "ConnectAngle" because we don't
    # want a rotation from 0 to 2pi to be weighted as a 2pi distance.
    ang_connected = ConnectAngle(angle[:], shape)

    ''' What velocity do we want? '''
    all_locations = np.vstack([gaussian_filter(position[:, 0], sigma=smoother),
                               gaussian_filter(position[:, 1], sigma=smoother),
                               gaussian_filter(ang_connected, sigma=smoother)
                               ])
    '''trajectory smoothed with Gaussian (over ' + str(second_smooth) + ' seconds)'''
    '''addition = addition + '\n only took component of velocity ' + str(args) + 'into account'''

    if 'x' in args:
        locations = np.vstack([locations, all_locations[0]])
        units.append(' cm/s in x')
    if 'y' in args:
        locations = np.vstack([locations, all_locations[1]])
        units.append(' cm/s in y')
    if 'angle' in args:
        locations = np.vstack([locations, all_locations[2]])
        units.append(' rad/s in angle')
    if len([x for x in ['x', 'y', 'angle'] if x in args]) == 0:  # if no components are specified
        locations = np.vstack([locations, all_locations])
        units = [' cm/s in x', ' cm/s in y', ' radian/s in angle']

    locations = locations[1:]

    velocities = np.zeros([locations.shape[0], locations.shape[1] - 1])

    ''' Calculate velocity '''
    for component in range(locations.shape[0]):
        for i in range(locations.shape[1] - 1):
            velocities[component, i] = (locations[component, i + 1] - locations[component, i]) * fps

    if 'withoutConnectors':
        pass
        # StartedScripts: add an option to the velocity to exclude the frames from the SmoothConnector

    if 'abs' in args:
        for ang_index in np.where(['radian/s' in unit for unit in units])[0]:
            velocities[ang_index] = velocities[ang_index] * average_radius(size, shape, solver)
        return [abs(np.linalg.norm(velocities[:, ii])) for ii in range(len(velocities[0]))]

    return velocities


def polar_velocity(position, angle, fps, size, shape, second_smooth):
    x_vel, y_vel = velocity(position, angle, fps, size, shape, second_smooth, 'x')[0], \
                   velocity(position, angle, fps, size, shape, second_smooth, 'y')[0]
    radius = np.sqrt(x_vel * x_vel + y_vel * y_vel)
    theta = [np.arctan(y_vel[i] / x_vel[i]) for i in range(radius.shape[0])]
    return radius, theta


def velocity_x(x, second_smooth, *args, **kwargs):
    return velocity(x.position, x.angle, x.fps, x.size, x.shape, second_smooth, x.solver, *args, **kwargs)


def acceleration(x, second_smooth, *args, **kwargs):
    """ If I add specifications to args, I only return these specific velocities,
    otherwise, I return x, y and angular velocity"""
    vel = velocity_x(x, second_smooth, *args, **kwargs)
    acc = np.zeros([vel.shape[0], vel.shape[1] - 1])

    ''' Calculate velocity '''
    for component in range(vel.shape[0]):
        for i in range(vel.shape[1] - 1):
            acc[component, i] = (vel[component, i + 1] - vel[component, i]) * x.fps

    if 'withoutConnectors':
        pass
        # StartedScripts: add an option to the velocity to exclude the frames from the SmoothConnector

    return acc


def check_for_false_tracking(x):
    vel = velocity_x(x, 0)
    lister = [x_vel or y_vel or ang_vel or isNaN for x_vel, y_vel, ang_vel, isNaN in
              zip(vel[0, :] > max_Vel_trans[x.size],
                  vel[1, :] > max_Vel_trans[x.size],
                  vel[2, :] > max_Vel_angle[x.size],
                  np.isnan(sum(vel[:]))
                  )]

    m = ranges(lister, 'boolean', scale=x.frames, smallestGap=20, buffer=8)
    # m = ranges(lister, 'boolean', smallestGap = 20, buffer = 4)
    print('False Tracking Regions: ' + str(m))
    return m


def plotVelocity(x):
    x_vel, y_vel = velocity_x(x, 1, 'x')[0], velocity_x(x, 1, 'y')[0]
    plt.plot(range(x_vel.shape[0]), x_vel)
    plt.plot(range(y_vel.shape[0]), y_vel)
    plt.show()


def velocity_distribution_plotting(x):
    x_vel, y_vel = velocity_x(x, 1, 'x')[0], velocity_x(x, 1, 'y')[0]
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(x_vel, bins=30)
    axs[1].hist(y_vel, bins=30)
    plt.show()
    hist_x, bin_edges_x = axs[0].hist(x_vel, bins=30)[:2]
    hist_y, bin_edges_y = axs[1].hist(y_vel, bins=30)[:2]

    return hist_x, bin_edges_x, hist_y, bin_edges_y


def velocity_distribution_fitting(x):
    x_vel, y_vel = velocity_x(x, 1, 'x')[0], velocity_x(x, 1, 'y')[0]
    hist, bin_edges = velocity_distribution_plotting(x)
    average = np.mean(x_vel)  # note this correction
    sigma = sum(bin_edges[:-1] * (hist - average) ** 2) / hist.shape[0]  # note this correction

    popt, pcov = curve_fit(hist, gauss, bin_edges[:-1], p0=[1, average, sigma])

    plt.plot(hist, bin_edges[:-1], 'b+:', label='data')
    plt.plot(hist, gauss(hist, *popt), 'ro:', label='fit')
    plt.legend()
    plt.title('Gaussian for x_speed')
    plt.xlabel('interval')
    plt.ylabel('number of frames')  # or 1/s that we spent in this state
    plt.show()
