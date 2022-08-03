from abc import abstractmethod
from PhysicsEngine.drawables import Drawables, colors
import pygame
import numpy as np


class Participants(Drawables):
    def __init__(self, x, color=colors['puller']):
        super().__init__(color)
        self.solver = x.solver
        self.filename = x.filename
        self.frames = list()
        self.size = x.size
        self.VideoChain = x.VideoChain
        self.positions = np.array([])

    @abstractmethod
    def matlab_loading(self, x) -> None:
        pass

    @abstractmethod
    def averageCarrierNumber(self) -> float:
        pass

    def draw(self, display) -> None:
        if self.solver == 'ant':
            for part in range(self.positions[display.i].shape[0]):
                pygame.draw.circle(display.screen, self.color, display.m_to_pixel(self.positions[display.i][part]), 7.)
        else:
            for part in range(self.positions[display.i].shape[0]):
                pygame.draw.circle(display.screen, self.color,
                                   display.m_to_pixel(self.positions[display.i][part, 0]), 7.)
