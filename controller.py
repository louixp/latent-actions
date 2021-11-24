from typing import Tuple

import pygame
import torch

class Controller:

    def __init__(self, scale: int = 1):
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.scale = scale

    def get_action(self) -> torch.Tensor:
        pygame.event.get()
        action = torch.tensor(
            [[self.joystick.get_axis(0), self.joystick.get_axis(1)]])
        return self.scale * action
