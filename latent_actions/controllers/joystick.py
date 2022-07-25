from typing import Tuple

import pygame
import torch


class ExitException(Exception):
    pass


class JoystickController:

    def __init__(self, 
            x_center: float, 
            y_center: float,
            x_scale: float,
            y_scale: float):
        
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.x_center = x_center
        self.y_center = y_center
        self.x_scale = x_scale
        self.y_scale = y_scale

    def get_action(self) -> torch.Tensor:
        pygame.event.get()
        
        if self.joystick.get_button(8):
            raise ExitException('Back button pushed. Exiting Simulation.')

        return torch.tensor([[
                self.joystick.get_axis(0) * self.x_scale + self.x_center, 
                self.joystick.get_axis(1) * self.y_scale + self.y_center]])


if __name__ == '__main__':
    joystick = JoystickController(1, 1, 1, 1).joystick
    print('Buttons\t', end='\t')
    print('Axes')

    while True:
        _ = pygame.event.get()
        for i in range(joystick.get_numbuttons()):
            print(joystick.get_button(i), end='')
        print(end='\t')

        for i in range(joystick.get_numaxes()):
            print(f'{joystick.get_axis(i):2f}', end='\t')

        print(end='\r') 
