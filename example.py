import multiprocessing as mp
import platform
from time import sleep

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import gym
import panda_gym

from cvae.cvae import ConditionalVAE
from controller import Controller

ACTION_SCALE = 10

def visualize(decoder, conn: mp.connection.Connection):
    def plot_function(i):
        x, y, z = np.meshgrid(np.arange(-10, 10, 1),
                              np.arange(-10, 10, 1),
                              0)
        
        latent_actions = np.concatenate((x, y), axis=-1)
        latent_actions = torch.from_numpy(latent_actions.reshape((-1, 2))).float() 
        latent_actions *= ACTION_SCALE
        context = conn.recv()        
        contexts = context.expand(latent_actions.shape[0], context.shape[1])

        decoded_actions = decoder(latent_actions, contexts)
        decoded_actions = decoded_actions.detach().numpy()[:, :3]
        decoded_actions = decoded_actions.reshape((x.shape[0], x.shape[1], 3))

        u, v, w = np.split(decoded_actions, 3, axis=-1) 

        ax.cla() 
        ax.quiver(x, y, z, u, v, w, length=0.01)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ani = FuncAnimation(fig, plot_function, interval=1000)
    plt.show()

def simulate(decoder, conn: mp.connection.Connection):
    controller = Controller(scale=ACTION_SCALE)

    env = gym.make('PandaPickAndPlace-v1', render=True).env
    obs = env.reset()

    done = False
    while not done:
        latent_action = controller.get_action()
        context = torch.from_numpy(obs['observation'])
        context = torch.unsqueeze(context, 0).float()
        conn.send(context)
        
        action = decoder(latent_action, context)
        action = action.detach().numpy()
        action = np.squeeze(action)

        obs, reward, done, info = env.step(action)
        sleep(0.1)

    env.close()

if __name__ == '__main__':
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')
    cvae = ConditionalVAE()
    conn_recv, conn_send = mp.Pipe(duplex=False)
    p_sim = mp.Process(target=simulate, args=(cvae.decoder, conn_send))
    p_viz = mp.Process(target=visualize, args=(cvae.decoder, conn_recv))
    p_sim.start()
    p_viz.start()
    p_sim.join()
    p_viz.join()

