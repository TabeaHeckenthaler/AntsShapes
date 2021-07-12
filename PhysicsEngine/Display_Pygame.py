import pygame
import numpy as np
from Setup.MazeFunctions import DrawGrid
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_DOWN, K_UP,
                           K_RIGHT, K_LEFT, K_r, K_l)
import math
import pygame.camera

global Delta_total, DeltaAngle_total
PPM, SCREEN_HEIGHT, screen = 0, 0, 0
Delta_total, DeltaAngle_total = [0, 0], 0

# printable colors
colors = {'my_maze': (0, 0, 0),
          'my_load': (250, 0, 0),
          'my_attempt_zone': (0, 128, 255),
          'text': (0, 0, 0),
          'background': (250, 250, 250),
          'background_inAttempt': (250, 250, 250),
          'contact': (51, 255, 51),
          'grid': (220, 220, 220),
          'arrow': (135, 206, 250),
          'participants': (0, 0, 0),
          }

pygame.font.init()  # display and fonts
font = pygame.font.Font('freesansbold.ttf', 25)


def Display_setup(my_maze, lines_stat, circles_stat, free=False):
    pygame.font.init()  # display and fonts
    pygame.font.Font('freesansbold.ttf', 25)
    global screen, PPM, SCREEN_HEIGHT

    if free:  # screen size dependent on trajectory
        print('free, I have a problem')
        # PPM = int(1000 / (np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10))  # pixels per meter
        # SCREEN_WIDTH = int((np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10) * PPM)
        # SCREEN_HEIGHT = int((np.max(x.position[:, 1]) - np.min(x.position[:, 1]) + 10) * PPM)

    else:  # screen size determined by maze size
        PPM = int(1500 / my_maze.arena_length)  # pixels per meter
        SCREEN_WIDTH, SCREEN_HEIGHT = 1500, int(my_maze.arena_height * PPM)

    # font = pygame.font.Font('freesansbold.ttf', 25)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    # 0 means default display, 32 is the depth
    # (something about colour and bits)
    pygame.display.set_caption(my_maze.shape + '  ' + my_maze.size + '  Ants solver')
    # what to print on top of the game window

    ''' Draw the fixtures which are not moving '''
    for body in my_maze.bodies:
        for fixture in body.fixtures:
            if not (body.userData == 'my_load'):
                if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
                    lines_stat.append([[(body.transform * v) for v in fixture.shape.vertices],
                                       colors[body.userData]])
                elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
                    circles_stat.append([fixture.shape.radius,
                                         body.position + fixture.shape.pos,
                                         colors[body.userData]])
    return lines_stat, circles_stat


def event_key(key, delta, delta_angle, lateral=0.05, rotational=0.01):
    if key == K_DOWN:
        delta = np.array(delta) + np.array([0, -lateral])
    elif key == K_UP:
        delta = np.array(delta) + np.array([0, lateral])
    elif key == K_RIGHT:
        delta = np.array(delta) + np.array([lateral, 0])
    elif key == K_LEFT:
        delta = np.array(delta) + np.array([-lateral, 0])
    elif key == K_r:
        delta_angle += rotational
    elif key == K_l:
        delta_angle -= rotational
    return list(delta), delta_angle


def Pygame_EventManager(i, my_maze, my_load, points, lines, circles, *args, **kwargs):
    from Setup.Load import Load_loop
    global Delta_total, DeltaAngle_total
    pause = False

    if 'pause' in kwargs:
        pause = kwargs['pause']

    points = points + [my_load.position]
    load_vertices = Load_loop(my_load)
    for ii in range(int(len(load_vertices) / 4)):
        lines.append([load_vertices[ii * 4: ii * 4 + 4], colors[my_load.userData]])  # Display the load

    Display_loop(i, my_maze, my_load, points, lines, circles, *args, **kwargs)
    events = pygame.event.get()

    for event in events:  # what happened in the last event?
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):  # you can also add 'or Finished'
            # The user closed the window or pressed escape
            Display_end('Pygame_EventManager')
            return False, i, False

        elif event.type == KEYDOWN and event.key == K_SPACE:
            pause = not pause

    if pause:
        # breakpoint()
        delta, delta_angle = [0, 0], 0

        for event in events:
            if hasattr(event, 'key'):
                delta, delta_angle = event_key(event.key, delta, delta_angle)
        if 'Trajectory' in kwargs.keys():
            x = kwargs['Trajectory']
            if delta != [0, 0] or delta_angle != 0:
                x.position = x.position + delta
                x.angle = x.angle + delta_angle
                x.x_error[0], x.y_error[0], x.angle_error[0] = x.x_error[0] + delta[0], x.y_error[0] + delta[1], \
                                                               x.angle_error[0] + delta_angle

        Delta_total, DeltaAngle_total = [arg1 + arg2 for arg1, arg2 in
                                         zip(Delta_total, delta)], DeltaAngle_total + delta_angle
        return True, i, pause

    return True, i, pause


def arrow(start, end, *args):
    rad = math.pi / 180
    start, end = [int(start[0] * PPM), SCREEN_HEIGHT - int(start[1] * PPM)], \
                 [int(end[0] * PPM), SCREEN_HEIGHT - int(end[1] * PPM)]
    thickness, trirad = int(0.1 * PPM), int(0.4 * PPM)
    pygame.draw.line(screen, colors['arrow'], start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
    arrow_width = 150
    pygame.draw.polygon(screen, colors['arrow'], ((end[0] + trirad * math.sin(rotation),
                                                   end[1] + trirad * math.cos(rotation)),
                                                  (end[0] + trirad * math.sin(rotation - arrow_width * rad),
                                                   end[1] + trirad * math.cos(rotation - arrow_width * rad)),
                                                  (end[0] + trirad * math.sin(rotation + arrow_width * rad),
                                                   end[1] + trirad * math.cos(rotation + arrow_width * rad))))

    for a in args:
        text = font.render(str(a), True, colors['text'])
        screen.blit(text, start)
    return


#
# scale_factor=1, color=(0, 0, 0)
# scale_factor=0.2,color=(1, 0, 0)

def Display_renew(i, my_maze, *args, interval=1, **kwargs):
    """
    :param int i: index of frame
    :param Maze my_maze: maze
    :param int interval: interval between two displayed frames
    """
    global screen, PPM, SCREEN_HEIGHT
    if 'wait' in kwargs.keys():
        pygame.time.wait(int(kwargs['wait']))

    if 'attempt' in kwargs and kwargs['attempt']:
        attempt = '_inAttempt'
    else:
        attempt = ''
    screen.fill(colors['background' + attempt])

    DrawGrid(screen, my_maze.arena_length, my_maze.arena_height, PPM, SCREEN_HEIGHT)
    if 'Trajectory' in kwargs.keys():
        x = kwargs['Trajectory']

        movie = x.old_filenames(0)
        if x.frames.size > 1:
            text = font.render(movie, True, colors['text'])
            text_rect = text.get_rect()
            text2 = font.render('Frame: ' + str(x.frames[i]), True, colors['text'])
            screen.blit(text2, [0, 25])
        else:
            text = font.render('Frame: ' + str(x.frames[0]), True, colors['text'])
            text_rect = text.get_rect()
        screen.blit(text, text_rect)


def Display_loop(i, my_maze, my_load, points, lines, circles, *args, free=False, **kwargs):
    # and draw all the circles passed (hollow, so I put two on top of each other)
    if "PhaseSpace" in kwargs.keys():
        if i < kwargs['interval']:
            kwargs['ps_figure'] = kwargs["PhaseSpace"].draw_trajectory(kwargs['ps_figure'],
                                                                       np.array([my_load.position[i]]),
                                                                       np.array([my_load.angle[i]]), scale_factor=1,
                                                                       color=(0, 0, 0))
        else:
            kwargs['ps_figure'] = kwargs["PhaseSpace"].draw_trajectory(kwargs['ps_figure'],
                                                                       my_load.position[i:i+kwargs['interval']],
                                                                       my_load.angle[i:i+kwargs['interval']],
                                                                       scale_factor=1,
                                                                       color=(1, 0, 0))

    if 'attempt' in kwargs and kwargs['attempt']:
        attempt = '_inAttempt'
    else:
        attempt = ''
    for circle in circles:
        pygame.draw.circle(screen, circle[2],
                           [int(circle[1][0] * PPM),
                            SCREEN_HEIGHT - int(circle[1][1] * PPM)], int(circle[0] * PPM),
                           )
        pygame.draw.circle(screen, colors['background' + attempt],
                           [int(circle[1][0] * PPM), SCREEN_HEIGHT - int(circle[1][1] * PPM)],
                           int(circle[0] * PPM) - 3
                           )

    # and draw all the lines passed
    for bodies in lines:
        line = [(line[0] * PPM, SCREEN_HEIGHT - line[1] * PPM) for line in bodies[0]]
        pygame.draw.lines(screen, bodies[1], True, line, 3)

    # and draw all the points passed
    for point in points:
        pygame.draw.circle(screen, colors['text'],
                           [int(point[0] * PPM), SCREEN_HEIGHT - int(point[1] * PPM)], 5)

    if not free:
        # pygame.draw.lines(screen, (250, 200, 0), True, (my_maze.zone)*PPM, 3)
        if 'contact' in kwargs:
            for contacts in kwargs['contact']:
                pygame.draw.circle(screen, colors['contact'],  # On the corner
                                   [int(contacts[0] * PPM),
                                    int(SCREEN_HEIGHT - contacts[1] * PPM)],
                                   10,
                                   )

    if 'forces' in kwargs:
        kwargs['arrows'] = []
        for arrow_function in kwargs['forces']:
            kwargs['arrows'] = kwargs['arrows'] + arrow_function(x, my_load, i)

    if 'arrows' in kwargs:
        for a_i in kwargs['arrows']:
            arrow(*a_i)

    if 'participants' in kwargs:
        for part in kwargs['participants'](x, my_load):
            pygame.draw.circle(screen, colors['participants'],
                               [int(part[0] * PPM), SCREEN_HEIGHT - int(part[1] * PPM)], 5)

    pygame.display.flip()
    return


def Display_end(filename):
    # global Delta_total
    # CreatePNG(pygame.display.get_surface(), filename)
    pygame.display.quit()
    # pygame.quit()
    return False


def CreatePNG(surface, filename, *args):
    pygame.image.save(surface, filename)
    if 'inlinePlotting' in args:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        img = mpimg.imread(filename)
        plt.imshow(img)
    return

# def CreateMovie():
#     if len(pygame.camera.list_cameras()) == 0:
#         pygame.camera.init()
#
#         # Ensure we have somewhere for the frames
#         try:
#             os.makedirs("Snaps")
#         except OSError:
#             pass
#
#         # cam = pygame.camera.Camera("/dev/video0", (640, 480))
#         cam = pygame.camera.Camera(0, (640, 480))
#         cam.start()
#
#         file_num = 0
#         done_capturing = False
#
#     while not done_capturing:
#         file_num = file_num + 1
#         image = cam.get_image()
#         screen.blit(image, (0, 0))
#         pygame.display.update()
#
#         # Save every frame
#         filename = "Snaps/%04d.png" % file_num
#         pygame.image.save(image, filename)
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done_capturing = True
#
#     # Combine frames to make video
#     os.system("avconv -r 8 -f image2 -i Snaps/%04d.png -y -qscale 0 -s 640x480 -aspect 4:3 result.avi")
