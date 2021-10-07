import pygame
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_DOWN, K_UP,
                           K_RIGHT, K_LEFT, K_r, K_l, K_d, K_a, K_KP4, K_KP6)
import math

# printable colors
colors = {'my_maze': (0, 0, 0),
          'my_load': (250, 0, 0),
          'my_attempt_zone': (0, 128, 255),
          'text': (0, 0, 0),
          'background': (250, 250, 250),
          'contact': (51, 255, 51),
          'grid': (220, 220, 220),
          'arrow': (135, 206, 250),
          'participants': (0, 0, 0),
          'puller': (0, 250, 0),
          'lifter': (0, 250, 0),
          'empty': (0, 0, 0),
          }


class Display:

    def __init__(self, my_maze, filename, wait=0):
        self.my_maze = my_maze
        self.filename = filename
        self.ppm = int(1500 / self.my_maze.arena_length)  # pixels per meter
        self.height = int(self.my_maze.arena_height * self.ppm)
        self.width = 1500

        pygame.font.init()  # display and fonts
        self.font = pygame.font.Font('freesansbold.ttf', 25)
        self.screen = self.create_screen()
        self.arrows = []
        self.wait = wait

    def create_screen(self, free=False, caption=str()):
        pygame.font.init()  # display and fonts
        pygame.font.Font('freesansbold.ttf', 25)

        if free:  # screen size dependent on trajectory_inheritance
            print('free, I have a problem')
            # TODO: fix the screen size of free trajectories
            # PPM = int(1000 / (np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10))  # pixels per meter
            # SCREEN_WIDTH = int((np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10) * PPM)
            # SCREEN_HEIGHT = int((np.max(x.position[:, 1]) - np.min(x.position[:, 1]) + 10) * PPM)

        screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        # 0 means default display, 32 is the depth
        # (something about colour and bits)
        if self.my_maze is not None:
            pygame.display.set_caption(self.my_maze.shape + ' ' + self.my_maze.size + ' ' + \
                                       self.my_maze.solver + ': ' + caption)
        # what to print on top of the game window
        return screen

    def update_screen(self, my_load, x, i, points=None, PhaseSpace=None, ps_figure=None, wait=0):
        if self.wait > 0:
            pygame.time.wait(int(self.wait))

        self.renew_screen(frame=x.frames[i], movie_name=x.filename)
        self.draw(my_load, x=x, i=i, points=points, PhaseSpace=PhaseSpace, ps_figure=ps_figure)
        end = self.keyboard_events()
        return end

    def renew_screen(self, frame=0, movie_name=None):
        self.screen.fill(colors['background'])

        self.drawGrid()

        if frame is not None:
            text = self.font.render(movie_name, True, colors['text'])
            text_rect = text.get_rect()
            text2 = self.font.render('Frame: ' + str(frame), True, colors['text'])
            self.screen.blit(text2, [0, 25])
            self.screen.blit(text, text_rect)

    @staticmethod
    def end_screen():
        pygame.display.quit()

    def pause_me(self):
        pygame.time.wait(int(100))
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN and event.key == K_SPACE:
                return
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.end_screen()
                return
        self.pause_me()

    def draw(self, my_load, free=False, x=None, **kwargs):

        # and draw all the circles passed (hollow, so I put two on top of each other)

        if not free:
            if 'contact' in kwargs:
                for contacts in kwargs['contact']:
                    pygame.draw.circle(self.screen, colors['contact'],  # On the corner
                                       [int(contacts[0] * self.ppm),
                                        int(self.height - contacts[1] * self.ppm)],
                                       10,
                                       )

        self.draw_bodies(my_load)
        self.draw_arrows()

        if 'participants' in kwargs:
            for part in kwargs['participants'](x, my_load):
                pygame.draw.circle(self.screen, colors['participants'],
                                   [int(part[0] * self.ppm), self.height - int(part[1] * self.ppm)], 5)

        pygame.display.flip()
        return

    def draw_bodies(self, my_load=None):
        lines, circles, points = self.lines_circles_points_of_bodies([*self.my_maze.bodies, my_load])
        for circle in circles:
            pygame.draw.circle(self.screen, circle[2],
                               [int(circle[1][0] * self.ppm),
                                self.height - int(circle[1][1] * self.ppm)], int(circle[0] * self.ppm),
                               )
            pygame.draw.circle(self.screen, colors['background'],
                               [int(circle[1][0] * self.ppm), self.height - int(circle[1][1] * self.ppm)],
                               int(circle[0] * self.ppm) - 3
                               )

        for bodies in lines:
            line = [(line[0] * self.ppm, self.height - line[1] * self.ppm) for line in bodies[0]]
            pygame.draw.lines(self.screen, bodies[1], True, line, 3)

        for point in points:
            pygame.draw.circle(self.screen, colors['text'],
                               [int(point[0] * self.ppm), self.height - int(point[1] * self.ppm)], 5)

    def draw_contacts(self, contact):
        for contacts in contact:
            pygame.draw.circle(self.screen, colors['contact'],  # On the corner
                               [int(contacts[0] * self.ppm),
                                int(self.height - contacts[1] * self.ppm)],
                               10,
                               )

    def draw_arrows(self):
        for a_i in self.arrows:
            start, end, name = a_i
            start = [int(start[0] * self.ppm), self.height - int(start[1] * self.ppm)]

            if end is None and name == 'lifter' or name == 'empty':
                pygame.draw.circle(self.screen, colors[name], start, 5)

            else:
                end = [int(end[0] * self.ppm), self.height - int(end[1] * self.ppm)]
                rad = math.pi / 180
                thickness, trirad = int(0.05 * self.ppm), int(0.2 * self.ppm)
                arrow_width = 150

                if name in ['puller', 'lifter', 'empty']:
                    color = colors[name]
                else:
                    color = colors['arrow']
                    text = self.font.render(str(name), True, colors['text'])
                    self.screen.blit(text, end)

                pygame.draw.line(self.screen, color, start, end, thickness)
                pygame.draw.circle(self.screen, color, start, 5)

                rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
                pygame.draw.polygon(self.screen, color, ((end[0] + trirad * math.sin(rotation),
                                                          end[1] + trirad * math.cos(rotation)),
                                                         (end[0] + trirad * math.sin(rotation - arrow_width * rad),
                                                          end[1] + trirad * math.cos(rotation - arrow_width * rad)),
                                                         (end[0] + trirad * math.sin(rotation + arrow_width * rad),
                                                          end[1] + trirad * math.cos(rotation + arrow_width * rad))))

    def drawGrid(self):
        block = 2
        block_size = 2 * self.ppm
        for y in range(np.int(np.ceil(self.height / self.ppm / block) + 1)):
            for x in range(np.int(np.ceil(self.width / self.ppm / block))):
                rect = pygame.Rect(x * block_size, self.height -
                                   y * block_size, block_size, block_size)
                pygame.draw.rect(self.screen, colors['grid'], rect, 1)

    def keyboard_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):  # you can also add 'or Finished'
                # The user closed the window or pressed escape
                self.end_screen()
                return True

            if event.type == KEYDOWN and event.key == K_SPACE:
                self.pause_me()
            # """
            # To control the frames:
            # 'D' = one frame forward
            # 'A' = one frame backward
            # '4' (one the keypad) = one second forward
            # '6' (one the keypad) = one second backward
            # """
            # if event.key == K_a:
            #     i -= 1
            # elif event.key == K_d:
            #     i += 1
            # elif event.key == K_KP4:
            #     i -= 30
            # elif event.key == K_KP6:
            #     i += 30

    @staticmethod
    def lines_circles_points_of_bodies(bodies):
        lines, circles, points = [], [], []
        for body in bodies:
            for fixture in body.fixtures:
                if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
                    lines.append([[(body.transform * v) for v in fixture.shape.vertices], colors[body.userData]])
                elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
                    circles.append([fixture.shape.radius, body.position + fixture.shape.pos, colors[body.userData]])

            if body.userData == 'my_load':
                points.append(np.array(body.position))
        return lines, circles, points

    def snapshot(self, surface, filename, *args):
        pygame.image.save(surface, filename)
        if 'inlinePlotting' in args:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.set_size_inches(30, 15)
            img = mpimg.imread(filename)
            plt.imshow(img)
        return

    def draw_PhaseSpace(self, PhaseSpace, ps_figure, x, i, my_load, interval):
        if i < interval:
            ps_figure = PhaseSpace.draw_trajectory(ps_figure,
                                                   np.array([my_load.position]),
                                                   np.array([my_load.angle]),
                                                   scale_factor=1,
                                                   color=(0, 0, 0))
        else:
            ps_figure = PhaseSpace.draw_trajectory(ps_figure,
                                                   x.position[i:i + interval],
                                                   x.angle[i:i + interval],
                                                   scale_factor=1,
                                                   color=(1, 0, 0))
