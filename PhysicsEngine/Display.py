import pygame
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_DOWN, K_UP, K_RIGHT, K_LEFT, K_r, K_l, K_d, K_a, K_KP4, K_KP6)
from PhysicsEngine.drawables import colors


class Display:
    def __init__(self, x, my_maze, wait=0):
        self.my_maze = my_maze
        self.filename = x.filename
        self.ppm = int(1500 / self.my_maze.arena_length)  # pixels per meter
        self.height = int(self.my_maze.arena_height * self.ppm)
        self.width = 1500

        pygame.font.init()  # display and fonts
        self.font = pygame.font.Font('freesansbold.ttf', 25)
        self.screen = self.create_screen()
        self.arrows = []
        self.circles = []
        self.polygons = []
        self.points = []
        self.wait = wait
        self.i = 0
        self.renew_screen()

    def create_screen(self, free=False, caption=str()) -> pygame.surface:
        pygame.font.init()  # display and fonts
        pygame.font.Font('freesansbold.ttf', 25)

        if free:  # screen size dependent on trajectory_inheritance
            print('free, I have a problem')
            # TODO: fix the screen size of free trajectories
            # PPM = int(1000 / (np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10))  # pixels per meter
            # SCREEN_WIDTH = int((np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10) * PPM)
            # SCREEN_HEIGHT = int((np.max(x.position[:, 1]) - np.min(x.position[:, 1]) + 10) * PPM)

        screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        if self.my_maze is not None:
            pygame.display.set_caption(self.my_maze.shape + ' ' + self.my_maze.size + ' ' + self.my_maze.solver + ': ' + caption)
        return screen

    def m_to_pixel(self, r):
        return [int(r[0] * self.ppm), self.height - int(r[1] * self.ppm)]

    def update_screen(self, x, i):
        self.i = i
        if self.wait > 0:
            pygame.time.wait(int(self.wait))
        self.draw(x)
        end = self.keyboard_events()
        return end

    def renew_screen(self, frame=0, movie_name=None):
        self.screen.fill(colors['background'])

        self.drawGrid()
        self.polygons = self.circles = self.points = self.arrows = []

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

    def draw(self, x):
        self.my_maze.draw(self)
        if hasattr(x, 'participants'):
            if hasattr(x.participants, 'forces'):
                x.participants.forces.draw(self, x)
            if hasattr(x.participants, 'positions'):
                x.participants.draw(self)
        self.display()
        return

    def display(self):
        pygame.display.flip()

    def draw_contacts(self, contact):
        for contacts in contact:
            pygame.draw.circle(self.screen, colors['contact'],  # On the corner
                               [int(contacts[0] * self.ppm),
                                int(self.height - contacts[1] * self.ppm)],
                               10,
                               )

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
