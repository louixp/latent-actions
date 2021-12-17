from typing import Tuple

import pygame
import torch


class ExitException(Exception):
    pass


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
        
        if self.joystick.get_button(8):
            raise ExitException('Back button pushed. Exiting Simulation.')

        axes = [self.joystick.get_axis(0), 
                self.joystick.get_axis(1), 
                self.joystick.get_axis(2), 
                self.joystick.get_axis(1)]
        action = torch.tensor([axes[:self.DoF]])
        return self.scale * action


if __name__ == '__main__':
    joystick = Controller().joystick
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
