import numpy as np
from scipy.ndimage import median_filter
from Setup.Load import getLoadDim, centerOfMass_shift


def States_loop(my_maze, x, i, **kwargs):
    return [getState(x.solver, x.shape, x.size, x.position[i], x.angle[i], my_maze) for index in range(kwargs['interval'])]


def getState(solver, shape, size, position, angle, maze):
    shape_thickness, shape_height, shape_width, short_edge = getLoadDim(solver, shape, size)
    state = 0
    """ Shape SPT """
    if shape == 'SPT':
        h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle part of the T. (1.445 d)
        displacement = shape_width / 2 + h
        slit0, slit1 = maze.slits[0], maze.slits[1]

        # This is in the bulk bifnim :) 
        s = position[0] + (shape_width - displacement + shape_thickness) * np.cos(angle)
        li = position[0] - (displacement + shape_thickness) * np.cos(angle)

        if li < slit0 and slit1 < s:  # and maze.possible_state_transitions(states[-1], states[-2]):
            state = maze.statenames[0]
        elif li < slit0 < s < slit1:
            state = maze.statenames[1]
        elif li < slit0 and s < slit0:
            state = maze.statenames[2]
        elif (slit0 < li) and s < slit0:
            state = maze.statenames[3]
        elif slit0 < li < slit1 and slit0 < s:
            state = maze.statenames[4]
        elif li > slit1 > s:
            state = maze.statenames[5]
        elif slit1 < li and slit1 < s:
            state = maze.statenames[6]
        else:
            print('We did not find a state...?')
            breakpoint()
    return state


def States(x, *vargs, **kwargs):
    from Setup.Attempts import smoothing_window
    window = x.fps * int(smoothing_window / 2)
    speed = 10
    x, states = x.play(speed, 'states', *vargs, **kwargs)[:len(x.frames)]  # why len()...

    states_smoothed = median_filter(states, size=window)
    '''Median filter with window ' + str(smoothing_window) + ' s, when separating states '''

    summary = [[states_smoothed[0], 0]]
    for index in range(len(states_smoothed)):
        if summary[-1][0] != states_smoothed[index]:
            summary[-1].append(index)
            summary.append([states_smoothed[index], index])
    summary[-1].append(len(states_smoothed))
    return summary

    # """ Shape H """
    # if x.shape == 'H':
    #     # This is in the bulk bifnim :) 
    #     if x_pos < maze.slits[0] - shape_height/2:
    #         state = maze.statenames[0]

    #     # This is already entering the slit, but not passing it with the center of mass
    #     elif x_pos < maze.slits[0]:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[1]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:   
    #             state = maze.statenames[2]
    #         else: ('We did not find a state...?')

    #     # The H has passed the slit with the center of mass, but still has to get the last force_vector out of the slit
    #     elif x_pos < maze.slits[0] + shape_height/2:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[3]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:
    #             state = maze.statenames[4]
    #         else: ('We did not find a state...?')

    #     # The H is not stuck in the slit, but has entered the bulk of bachutz :) 
    #     else:
    #         state = maze.statenames[5]

    # """ Shape I """
    # if x.shape == 'I':
    #     # This is in the bulk bifnim :) 
    #     if x_pos < maze.slits[0] - shape_height/2:
    #         state = maze.statenames[0]

    #     # This is already entering the slit, but not passing it with the center of mass
    #     elif x_pos < maze.slits[0]:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[1]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:   
    #             state = maze.statenames[2]
    #         else: ('We did not find a state...?')

    #     # The H has passed the slit with the center of mass, but still has to get the last force_vector out of the slit
    #     elif x_pos < maze.slits[0] + shape_height/2:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[3]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:
    #             state = maze.statenames[4]
    #         else: ('We did not find a state...?')

    #     # The H is not stuck in the slit, but has entered the bulk of bachutz :) 
    #     else:
    #         state = maze.statenames[5]

    # """ Shape T """
    # if x.shape == 'T':
    #     # This is in the bulk bifnim :) 
    #     if x_pos < maze.slits[0] - shape_height/2:
    #         state = maze.statenames[0]

    #     # This is already entering the slit, but not passing it with the center of mass
    #     elif x_pos < maze.slits[0]:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[1]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:   
    #             state = maze.statenames[2]
    #         else: ('We did not find a state...?')

    #     # The H has passed the slit with the center of mass, but still has to get the last force_vector out of the slit
    #     elif x_pos < maze.slits[0] + shape_height/2:
    #         if np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 1:
    #             state = maze.statenames[3]
    #         elif np.mod(ang/(2*pi)*4 - np.mod(ang/(2*pi)*4, 1), 2) == 0:
    #             state = maze.statenames[4]
    #         else: ('We did not find a state...?')

    #     # The H is not stuck in the slit, but has entered the bulk of bachutz :) 
    #     else:
    #         state = maze.statenames[5]


shapes = ['SPT']


