import pygame
import numpy as np
from Setup.MazeFunctions import DrawGrid
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_DOWN, K_UP,
                           K_RIGHT, K_LEFT, K_r, K_l, K_d, K_a, K_KP4, K_KP6)
import math
import pygame.camera
from Setup.Load import Loops

global Delta_total, DeltaAngle_total
PPM, SCREEN_HEIGHT, SCREEN_WIDTH = 0, 0, 0
Delta_total, DeltaAngle_total = [0, 0], 0
global flag

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
          'puller': (0, 250, 0),
          'lifter': (0, 250, 0),
          'empty': (0, 0, 0),
          }

pygame.font.init()  # display and fonts
font = pygame.font.Font('freesansbold.ttf', 25)


def lines_circles_points(my_maze, my_load=None, lines=None, circles=None, points=None):
    if lines is None:
        lines = []
    if circles is None:
        circles = []
    if points is None:
        points = []

    for body in my_maze.bodies:
        body_lines(body, lines=lines, circles=circles, points=points)

    if my_load is not None:
        body_lines(my_load, lines=lines, circles=circles, points=points)

    return lines, circles, points


def body_lines(body, lines, circles, points):
    for fixture in body.fixtures:
        if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
            lines.append([[(body.transform * v) for v in fixture.shape.vertices], colors[body.userData]])
        elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
            circles.append([fixture.shape.radius, body.position + fixture.shape.pos, colors[body.userData]])

    if body.userData == 'my_load':
        points.append(np.array(body.position))


def Display_screen(my_maze=None, free=False, caption=None):
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
    if my_maze is not None:
        pygame.display.set_caption(my_maze.shape + ' ' + my_maze.size + ' ' + my_maze.solver + ': ' + caption)
    # what to print on top of the game window
    return screen


def event_key(key, delta, delta_angle, i, lateral=0.05, rotational=0.01):
    """
    To control the frames:
    'D' = one frame forward
    'A' = one frame backward
    '4' (one the keypad) = one second forward
    '6' (one the keypad) = one second backward
    """

    # if key == K_DOWN:
    #     delta = np.array(delta) + np.array([0, -lateral])
    # elif key == K_UP:
    #     delta = np.array(delta) + np.array([0, lateral])
    # elif key == K_RIGHT:
    #     delta = np.array(delta) + np.array([lateral, 0])
    # elif key == K_LEFT:
    #     delta = np.array(delta) + np.array([-lateral, 0])
    # elif key == K_r:
    #     delta_angle += rotational
    # elif key == K_l:
    #     delta_angle -= rotational
    if key == K_a:
        i -= 1
    elif key == K_d:
        i += 1
    elif key == K_KP4:
        i -= 30
    elif key == K_KP6:
        i += 30
    return list(delta), delta_angle, i


def Pygame_EventManager(x, i, my_load, my_maze, screen, points=None, arrows=None,
                        PhaseSpace=None, ps_figure=None,
                        wait=0, **kwargs):
    global Delta_total, DeltaAngle_total
    pause = False

    # TODO: install a better pause!!!

    if 'pause' in kwargs:
        pause = kwargs['pause']

    Display_renew(screen, my_maze=my_maze, i=i, frame=x.frames[i], movie_name=x.old_filenames(0), wait=wait, **kwargs)
    Display_loop(my_load, my_maze, screen, x=x, i=i, points=points, arrows=arrows,
                 PhaseSpace=PhaseSpace, ps_figure=ps_figure, **kwargs)
    events = pygame.event.get()

    for event in events:  # what happened in the last event?
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):  # you can also add 'or Finished'
            # The user closed the window or pressed escape
            Display_end()
            return False, i, False

        elif event.type == KEYDOWN and event.key == K_SPACE:
            pause = not pause

    if pause:
        delta, delta_angle = [0, 0], 0

        for event in events:
            if hasattr(event, 'key') and event.type == KEYDOWN:
                pygame.key.set_repeat(500, 100)
                delta, delta_angle, i = event_key(event.key, delta, delta_angle, i)

            if delta != [0, 0] or delta_angle != 0:
                x.position = x.position + delta
                x.angle = x.angle + delta_angle
                x.x_error[0], x.y_error[0], x.angle_error[0] = x.x_error[0] + delta[0], x.y_error[0] + delta[1], \
                                                               x.angle_error[0] + delta_angle

        Delta_total, DeltaAngle_total = [arg1 + arg2 for arg1, arg2 in
                                         zip(Delta_total, delta)], DeltaAngle_total + delta_angle

        return True, i, pause

    return True, i, pause


def arrow(start, end, name, screen):
    start = [int(start[0] * PPM), SCREEN_HEIGHT - int(start[1] * PPM)]

    if end is None and name == 'lifter' or name == 'empty':
        pygame.draw.circle(screen, colors[name], start, 5)
        return

    end = [int(end[0] * PPM), SCREEN_HEIGHT - int(end[1] * PPM)]
    rad = math.pi / 180
    thickness, trirad = int(0.05 * PPM), int(0.2 * PPM)
    arrow_width = 150

    if name in ['puller', 'lifter', 'empty']:
        color = colors[name]
    else:
        color = colors['arrow']
        text = font.render(str(name), True, colors['text'])
        screen.blit(text, end)

    pygame.draw.line(screen, color, start, end, thickness)
    pygame.draw.circle(screen, color, start, 5)

    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
    pygame.draw.polygon(screen, color, ((end[0] + trirad * math.sin(rotation),
                                         end[1] + trirad * math.cos(rotation)),
                                        (end[0] + trirad * math.sin(rotation - arrow_width * rad),
                                         end[1] + trirad * math.cos(rotation - arrow_width * rad)),
                                        (end[0] + trirad * math.sin(rotation + arrow_width * rad),
                                         end[1] + trirad * math.cos(rotation + arrow_width * rad))))
    return


def Display_renew(screen, my_maze=None, frame=None, movie_name=None, wait=0, **kwargs):
    """

    :param screen:
    :param my_maze:
    :param frame:
    :param movie_name:
    :param wait:
    :param kwargs:
    :return:
    """
    if wait > 0:
        pygame.time.wait(int(wait))

    if 'attempt' in kwargs and kwargs['attempt']:
        attempt = '_inAttempt'
    else:
        attempt = ''
    screen.fill(colors['background' + attempt])
    if my_maze is not None:
        global PPM, SCREEN_HEIGHT
        DrawGrid(screen, my_maze.arena_length, my_maze.arena_height, PPM, SCREEN_HEIGHT)

    if frame is not None:
        text = font.render(movie_name, True, colors['text'])
        text_rect = text.get_rect()
        text2 = font.render('Frame: ' + str(frame), True, colors['text'])
        screen.blit(text2, [0, 25])
        screen.blit(text, text_rect)


def Display_loop(my_load, my_maze, screen, free=False, x=None, i=None, lines=None, circles=None, points=None,
                 arrows=None, PhaseSpace=None, ps_figure=None, **kwargs):
    if lines is None:
        lines = []
    if circles is None:
        circles = []
    if points is None:
        points = []
    if arrows is None:
        arrows = []

    lines, circles, points = lines_circles_points(my_maze, lines=lines, circles=circles, points=points)

    # and draw all the circles passed (hollow, so I put two on top of each other)
    if PhaseSpace is not None:
        if i < kwargs['interval']:
            ps_figure = PhaseSpace.draw_trajectory(ps_figure,
                                                   np.array([my_load.position]),
                                                   np.array([my_load.angle]),
                                                   scale_factor=1,
                                                   color=(0, 0, 0))
        else:
            ps_figure = PhaseSpace.draw_trajectory(ps_figure,
                                                   x.position[i:i + kwargs['interval']],
                                                   x.angle[i:i + kwargs['interval']],
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

    [arrow(*a_i, screen) for a_i in arrows]

    if 'arrows' in kwargs:
        for a_i in kwargs['arrows']:
            arrow(*a_i, screen)

    if 'participants' in kwargs:
        for part in kwargs['participants'](x, my_load):
            pygame.draw.circle(screen, colors['participants'],
                               [int(part[0] * PPM), SCREEN_HEIGHT - int(part[1] * PPM)], 5)

    pygame.display.flip()
    return


def Display_end(filename=None):
    # global Delta_total

    if filename is not None:
        CreatePNG(pygame.display.get_surface(), filename)

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
