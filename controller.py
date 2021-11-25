from typing import Tuple

import pygame
import torch

class Controller:

    def __init__(self, scale: int = 1, DoF: int = 2):
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.scale = scale
        self.DoF = DoF

    def get_action(self) -> torch.Tensor:
        pygame.event.get()
        axes = [self.joystick.get_axis(0), 
                self.joystick.get_axis(1), 
                self.joystick.get_axis(2), 
                self.joystick.get_axis(1)]
        action = torch.tensor([axes[:self.DoF]])
        return self.scale * action
