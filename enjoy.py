from argparse import ArgumentParser
import multiprocessing as mp
import platform
from time import sleep 
from typing import Iterable

import numpy as np
import torch
import gym
import panda_gym
from frankapy import FrankaArm

from latent_actions.cvae import vae, cvae, cae, gbc, aligned_decoder 
from latent_actions.controllers.joystick import JoystickController
from latent_actions.data.pick_and_place import PickAndPlaceDemonstrationDataset
from latent_actions import visualization


def find_latent_window(data_path: str, decoder: vae.VAE):
    dataset = PickAndPlaceDemonstrationDataset(
            data_path, include_goal=False, include_joint_angles=False, 
            dof=decoder.action_dim - 1, keep_success=False)
    
    x_max, y_max = -np.inf, -np.inf
    x_min, y_min = np.inf, np.inf

    for context, action in dataset:
        x = torch.cat((context, action)).reshape(1, -1)
        z = decoder.encoder(x)
        if not hasattr(decoder, 'fc_mu'):
            # AE with no mean and variance
            upper_bound = lower_bound = z.squeeze()
        else:
            # VAE with mean and variance
            mu = decoder.fc_mu(x)
            log_var = decoder.fc_var(x)
            std = torch.exp(log_var / 2)
            
            upper_bound = (mu + std).unsqueeze()
            lower_bound = (mu - std).unsqueeze()

        x_max = max(x_max, upper_bound[0])
        y_max = max(y_max, upper_bound[1])
        x_min = min(x_min, lower_bound[0])
        y_min = min(y_min, lower_bound[1])
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_scale = (x_max - x_min) / 2
    y_scale = (y_max - y_min) / 2

    return {
            'x_center': x_center.item(), 
            'y_center': y_center.item(),
            'x_scale': x_scale.item(),
            'y_scale': y_scale.item()}

def deploy_real_arm(
        decoder: vae.VAE, 
        controller: JoystickController):
    fa = FrankaArm()
    
    while True:
        try:
            latent_action = controller.get_action()
        except Exception as e:
            print('Exiting...')
            return 
        
        curr_pose = fa.get_pose()
        curr_joints = fa.get_joints()
        
        action = decoder(latent=torch.tensor(latent_action), context=None)
        action = np.squeeze(action.detach().numpy())

        curr_pose.from_frame = 'world'
        curr_pose.rotation = np.eye(3)
        fa.goto_joints(
                curr_joints + action, duration=3, ignore_virtual_walls=True)

def simulate(
        decoder: vae.VAE, 
        controller: JoystickController,
        conns: Iterable[mp.connection.Connection],
        step_rate: float,
        env_id: str):
    env = gym.make(env_id, render=True).env
    obs = env.reset()

    done = False
    while not done:
        try:
            latent_action = controller.get_action()
        except Exception as e:
            print('Simulation exiting...')
            for conn in conns:
                conn.send(None)
            return

        context = torch.from_numpy(obs['observation'])
        if decoder.hparams.get('include_goal'):
            goal = torch.from_numpy(obs['desired_goal'])
            context = torch.cat((context, goal))
        if decoder.hparams.get('include_joint_angles'):
            joint_angles = torch.tensor([
                    env.sim.get_joint_angle(env.robot.body_name, i)
                    for i in range(7)])
            context = torch.cat((context, joint_angles))

        context = torch.unsqueeze(context, 0).float()
        for conn in conns:
            conn.send((latent_action, context))
        
        action = decoder(latent=latent_action, context=context)
        action = action.detach().numpy()
        action = np.squeeze(action)

        obs, reward, done, info = env.step(action)
        sleep(step_rate)

    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: Figure out a better way to write this.
    parser.add_argument(
            '--deploy_target', type=str, required=True, choices=['real', 'sim'])
    parser.add_argument(
            '--model_class', default='VAE', type=str, 
            choices=['VAE', 'cVAE', 'cAE', 'gBC', 'align'])
    parser.add_argument(
            '--checkpoint_path', default=None, type=str, required=True)
    parser.add_argument('--data_path', default=None, type=str, required=True)
    parser.add_argument('--step_rate', default=0.1, type=float)
    args = parser.parse_args()

    if args.model_class == 'VAE':
        ModelClass = vae.VAE
    elif args.model_class == 'gBC':
        ModelClass = gbc.GaussianBC
    elif args.model_class == 'cVAE':
        ModelClass = cvae.ConditionalVAE
    elif args.model_class == 'cAE':
        ModelClass = cae.ConditionalAE
    elif args.model_class == 'align':
        ModelClass = aligned_decoder.AlignedDecoder
    
    decoder = ModelClass.load_from_checkpoint(args.checkpoint_path)
    print('Decoder loaded.')
    
    latent_window = find_latent_window(args.data_path, decoder)
    print(f'Latent window: {latent_window}')
    
    controller = JoystickController(**latent_window)

    if args.deploy_target == 'real':
        deploy_real_arm(decoder, controller)

    if decoder.action_dim == 8:
        simulate(
                decoder=decoder, 
                conns=[], 
                step_rate=args.step_rate, 
                env_id='PandaPickAndPlaceJoints-v2',
                **latent_window)

    elif decoder.action_dim == 4:
        if platform.system() == 'Darwin':
            mp.set_start_method('spawn')

        conn_recv_1, conn_send_1 = mp.Pipe(duplex=False)
        conn_recv_2, conn_send_2 = mp.Pipe(duplex=False)

        p_sim = mp.Process(
                target=simulate, 
                args=(
                    decoder, 
                    [conn_send_1, conn_send_2], 
                    latent_window['x_center'],
                    latent_window['y_center'],
                    latent_window['x_scale'],
                    latent_window['y_scale'],
                    args.step_rate, 
                    'PandaPickAndPlace-v1'))
        p_viz_vec = mp.Process(
                target=visualization.visualize_latent_actions_in_3d, 
                args=(
                    decoder, 
                    conn_recv_1, 
                    visualization.plot_vector_field, 
                    latent_window['x_center'],
                    latent_window['y_center'],
                    latent_window['x_scale'],
                    latent_window['y_scale'],
                    5))
        p_viz_man = mp.Process(
                target=visualization.visualize_latent_actions_in_3d, 
                args=(
                    decoder, 
                    conn_recv_2, 
                    visualization.plot_manifold,
                    latent_window['x_center'],
                    latent_window['y_center'],
                    latent_window['x_scale'],
                    latent_window['y_scale'],
                    10))

        p_sim.start()
        p_viz_vec.start()
        p_viz_man.start()
        p_sim.join()
        p_viz_vec.join()
        p_viz_man.join()
