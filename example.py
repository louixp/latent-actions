from argparse import ArgumentParser
import multiprocessing as mp
import platform
from time import sleep 
from typing import Iterable

import numpy as np
import torch
import gym
import panda_gym

from cvae import cvae 
from controller import Controller
import visualization


def simulate(
        decoder: cvae.VAE, 
        conns: Iterable[mp.connection.Connection],
        action_scale: int,
        step_rate: float):
    controller = Controller(scale=action_scale)

    env = gym.make('PandaPickAndPlace-v1', render=True).env
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
    parser.add_argument(
            '--model_class', default='VAE', type=str, choices=['VAE', 'cVAE'])
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--action_scale', default=10, type=int)
    parser.add_argument('--step_rate', default=0.1, type=float)
    args = parser.parse_args()

    if args.model_class == 'VAE':
        ModelClass = cvae.VAE
    else:
        ModelClass = cvae.ConditionalVAE
    
    if args.checkpoint_path is not None:
        decoder = ModelClass.load_from_checkpoint(args.checkpoint_path)
        print('Decoder loaded.')
    else:
        decoder = ModelClass()
        print('Random decoder instantiated.')
    
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')

    conn_recv_1, conn_send_1 = mp.Pipe(duplex=False)
    conn_recv_2, conn_send_2 = mp.Pipe(duplex=False)

    p_sim = mp.Process(
            target=simulate, 
            args=(
                decoder, [conn_send_1, conn_send_2], args.action_scale, 
                args.step_rate))
    p_viz_vec = mp.Process(
            target=visualization.visualize_latent_actions_in_3d, 
            args=(decoder, conn_recv_1, visualization.plot_vector_field, 
                args.action_scale, 5))
    p_viz_man = mp.Process(
            target=visualization.visualize_latent_actions_in_3d, 
            args=(decoder, conn_recv_2, visualization.plot_manifold,
                args.action_scale, 10))

    p_sim.start()
    p_viz_vec.start()
    p_viz_man.start()
    p_sim.join()
    p_viz_vec.join()
    p_viz_man.join()
