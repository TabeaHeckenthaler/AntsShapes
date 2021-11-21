import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_DOWN, K_UP, K_RIGHT, K_LEFT, K_r, K_l, K_d, K_a, K_KP4, K_KP6)
from PhysicsEngine.drawables import colors
import os
import numpy as np
import cv2
from mss import mss
from PIL import Image
import sys
from Directories import video_directory


class Display:
    def __init__(self, x, my_maze, wait=0, ps=None, i=0, videowriter=True):
        self.my_maze = my_maze
        self.filename = x.filename
        self.ppm = int(1500 / self.my_maze.arena_length)  # pixels per meter
        self.height = int(self.my_maze.arena_height * self.ppm)
        self.width = 1500

        pygame.font.init()  # display and fonts
        self.font = pygame.font.Font('freesansbold.ttf', 25)
        # self.monitor = {'left': 0, 'top': 0,
        #                 'width': int(Tk().winfo_screenwidth() * 0.9), 'height': int(Tk().winfo_screenheight() * 0.8)}
        self.monitor = {'left': 0, 'top': 0,
                        'width': self.width, 'height': self.height}
        self.screen = self.create_screen(x)
        self.arrows = []
        self.circles = []
        self.polygons = []
        self.points = []
        self.wait = wait
        self.i = i
        my_maze.set_configuration(x.position[i], x.angle[i])
        self.renew_screen()
        self.ps = ps
        if videowriter:
            self.VideoWriter = cv2.VideoWriter(video_directory + sys.argv[0].split('/')[-1].split('.')[0] + '.mp4v',
                                               cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                               (self.monitor['width'], self.monitor['height']))
        k = 1

    def create_screen(self, x, caption=str()) -> pygame.surface:
        pygame.font.init()  # display and fonts
        pygame.font.Font('freesansbold.ttf', 25)

        if hasattr(x, 'free') and x.free:  # screen size dependent on trajectory_inheritance
            self.ppm = int(1000 / (np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10))  # pixels per meter
            self.width = int((np.max(x.position[:, 0]) - np.min(x.position[:, 0]) + 10) * self.ppm)
            self.height = int((np.max(x.position[:, 1]) - np.min(x.position[:, 1]) + 10) * self.ppm)

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.monitor['left'], self.monitor['top'])
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
            text2 = self.font.render('Frame: ' + str(self.i), True, colors['text'])
            self.screen.blit(text2, [0, 25])
            self.screen.blit(text, text_rect)

    def end_screen(self):
        if hasattr(self, 'VideoWriter'):
            self.VideoWriter.release()
        # if self.ps is not None:
        #     self.ps.VideoWriter.release()
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
        if self.ps is not None:
            if self.i <= 1 or self.i >= len(x.angle)-1:
                kwargs = {'color': (0, 0, 0), 'scale_factor': 1.}
            else:
                kwargs = {}
            self.ps.draw_trajectory(x.position[self.i:self.i+1], x.angle[self.i:self.i+1], **kwargs)
        if hasattr(x, 'participants'):
            if hasattr(x.participants, 'forces'):
                x.participants.forces.draw(self, x)
            if hasattr(x.participants, 'positions'):
                x.participants.draw(self)
        self.display()
        self.write_to_Video()
        return

    def display(self):
        pygame.display.flip()

    def write_to_Video(self):
        with mss() as sct:
            screenShot = sct.grab(self.monitor)
            img = Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb)
        # self.VideoWriter.write(pygame.surfarray.pixels3d(self.screen))
        if hasattr(self, 'VideoWriter'):
            self.VideoWriter.write(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        # if self.ps is not None:
        #     self.ps.write_to_Video()

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

    def snapshot(self, filename, *args):
        pygame.image.save(self.screen, filename)
        if 'inlinePlotting' in args:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.set_size_inches(30, 15)
            img = mpimg.imread(filename)
            plt.imshow(img)
        return



