import sys
sys.path.insert(0, '/home/c3po/frankapy/examples/latent-actions')
from latent_actions.data.dataset import Step, Episode, EpisodicDataset
import argparse
import time
from frankapy import FrankaArm
import pickle as pkl
from controller import Controller
import numpy as np
import os
import pygame
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=1000000000000000)
    parser.add_argument('--open_gripper', '-o', action='store_true')
    parser.add_argument('--file', '-f', default='franka_centerout_traj.pkl')
    args = parser.parse_args()
    controller = Controller(DoF=4)

    print('Starting robot')
    fa = FrankaArm()
    # fa.reset_pose()
    if args.open_gripper:
        fa.open_gripper()
    print('Applying 0 force torque control for {}s'.format(args.time))
    end_effector_position = []
    end_effector_velocity = []
    joint_position = []
    joint_velocity = []
    episode_begin = []
    episodic_dataset = EpisodicDataset()

    eepos_int = []
    jpos_int = []
    jvel_int = []
    gvel_int = []
    print("press and hold A to start recording trajectory, press B to finish recording and exit", end='\n')

    fa.run_guide_mode(args.time, block=False)
    done = False
    joystick = Controller().joystick
    print('Buttons\t', end='\t')
    print('Axes')
    started = False
    prev_ee_pos = 0
    prev_grip_pos = 0

    while not done:
        _ = pygame.event.get()

        t = time.time()
        if controller.joystick.get_button(1):
            episode = Episode()
            started = True
            # eepos_int.append(fa.get_pose())
            # jpos_int.append(fa.get_joints())
            # jvel_int.append(fa.get_joint_velocities())

            curr_ee_pos = fa.get_pose()
            curr_grip_pos = fa.get_gripper_width()

            step = Step(
                joint_velocity=fa.get_joint_velocities(),
                ee_velocity=curr_ee_pos-prev_ee_pos,
                context={'joint_angles': fa.get_joints(),
                         'gripper_width': curr_grip_pos},
                gripper_velocity=curr_grip_pos-prev_grip_pos
            )
            episode.append(step)
            prev_ee_pos = curr_ee_pos
            prev_grip_pos = curr_grip_pos

            # elapsed = time.time() - t
            # print(fa.get_pose(), '\r')
            # print('\n')
            print("A button pressed, record trajectory", end='\r')
            time.sleep(0.01)
        # print(eepos_int)
        # print(jpos_int)
        # print(jvel_int)
        if ((started == True) and not controller.joystick.get_button(1)):
            # end_effector_position.append(eepos_int)
            # joint_position.append(jpos_int)
            # joint_velocity.append(jvel_int)
            episodic_dataset.append(episode)
            # print(eepos_int)
            # print(jpos_int)
            # print(jvel_int)
            prev_ee_pos = 0

        if not controller.joystick.get_button(1):
            for i in range(controller.joystick.get_numbuttons()):
                print(joystick.get_button(i), end='')
            print(end='\t')
            for i in range(controller.joystick.get_numaxes()):
                print(f'{controller.joystick.get_axis(i):2f}', end='\t')
            print(end='\r')
            started = False
            # eepos_int = []
            # jpos_int = []
            # jvel_int = []

        if controller.joystick.get_button(2):
            done = True
            print('\n')
            print("finished recording trajectory. press back button to exit script")

    # end_effector_velocity = np.array(end_effector_position[1:]) - np.array(end_effector_position[:-1])

    # centerout_dict = {"ee_pos": end_effector_position, "ee_vel": end_effector_velocity, "j_pos": joint_position, "j_vel": joint_velocity}
    # pkl.dump(centerout_dict, open(args.file, 'wb'))
    # with open('franka_centerout_traj.pkl', "rb") as fp:
    #     data = pkl.load(fp)
    #     ee_pos = data["ee_pos"]
    #     ee_vel = data["ee_vel"]
    #     j_pos = data["j_pos"]
    #     j_vel = data["j_vel"]
    # print(ee_pos, "eepos")
    # print(ee_vel, "eevel")
    # print(j_pos, "jpos")
    # print(j_vel, "jvel")

    episodic_dataset.dump(args.file)
