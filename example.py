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

DEBUG = False 
ACTION_SCALE = 10

def visualize(decoder, conn: mp.connection.Connection):
    x, y, z = np.meshgrid(np.arange(-ACTION_SCALE, ACTION_SCALE, 2),
                          np.arange(-ACTION_SCALE, ACTION_SCALE, 2),
                          0)
    latent_actions = np.concatenate((x, y), axis=-1)
    latent_actions = torch.from_numpy(latent_actions.reshape((-1, 2))).float() 
    latent_actions *= ACTION_SCALE

    def plot_function(i):
        ax.cla() 

        prev_action, context = conn.recv()        
        prev_action = prev_action.numpy()[0]
        contexts = context.expand(latent_actions.shape[0], context.shape[1])

        decoded_actions = decoder(latent_actions, contexts)
        decoded_actions = decoded_actions.detach().numpy()[:, :3]
        decoded_actions = decoded_actions.reshape((x.shape[0], x.shape[1], 3))

        u, v, w = np.split(decoded_actions, 3, axis=-1) 
        ax.quiver(x, y, z, u, v, w, normalize=True)

        if DEBUG:
            decoded_actions_ref = decoder(
                    torch.zeros_like(latent_actions), 
                    torch.zeros_like(contexts))
            decoded_actions_ref = decoded_actions_ref.detach().numpy()[:, :3]
            decoded_actions_ref = decoded_actions_ref.reshape(
                    (x.shape[0], x.shape[1], 3))
            u_ref, v_ref, w_ref = np.split(decoded_actions_ref, 3, axis=-1) 
            ax.quiver(x, y, z, u_ref, v_ref, w_ref, color='lime', normalize=True)

        # Controller/latent current posistion.
        ax.plot(*prev_action, 'ro')
        # Small hacks to 'equalize' axes since matplotlib doesn't support it.
        ax.plot(0, 0, 2.5, alpha=0)
        ax.plot(0, 0, -2.5, alpha=0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Latent x')
    ax.set_ylabel('Latent y')
    ax.set_title('Latent Action Decoder Vector Field')
    ani = FuncAnimation(fig, plot_function, interval=1)
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
        conn.send((latent_action, context))
        
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

