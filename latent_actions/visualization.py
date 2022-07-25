import multiprocessing as mp
import sys
from typing import Callable

from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from cvae import vae 


def visualize_latent_actions_in_3d(
        decoder: vae.VAE, 
        conn: mp.connection.Connection,
        plot_function: Callable[
            [int, vae.VAE, mp.connection.Connection, matplotlib.axes.Axes, 
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
            None],
        x_center: float, 
        y_center: float,
        x_scale: float,
        y_scale: float,
        grid_size: int):
    x_step = x_scale / grid_size
    y_step = y_scale / grid_size
    x, y, z = np.meshgrid(np.arange(-x_scale, x_scale, x_step),
                          np.arange(-x_scale, y_scale, y_step),
                          0)
    latent_actions = np.concatenate((x, y), axis=-1)
    latent_actions = torch.from_numpy(latent_actions.reshape((-1, 2))).float() 
    latent_actions[:, 0] += x_center
    latent_actions[:, 1] += y_center
    norm = np.linalg.norm(latent_actions, axis=-1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ani = FuncAnimation(
            fig, plot_function, 
            fargs=(decoder, conn, ax, latent_actions, x, y, z, norm, 
                x_scale),
            interval=1)
    plt.show()


def plot_manifold(
        frame: int, 
        decoder: vae.VAE, 
        conn: mp.connection.Connection,
        ax: matplotlib.axes.Axes, 
        latent_actions: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray, c: np.ndarray,
        action_scale: int):
    ax.cla() 
    
    try:
        prev_action, context = conn.recv()        
    except:
        sys.exit()

    contexts = context.expand(latent_actions.shape[0], context.shape[1])
    decoded_actions = _decode_latent_actions(
            decoder, latent_actions, contexts, x)
    u, v, w = np.split(decoded_actions, 3, axis=-1) 
    sc = ax.scatter(u, v, w, c=c)


def plot_vector_field(
        frame: int, 
        decoder: vae.VAE, 
        conn: mp.connection.Connection,
        ax: matplotlib.axes.Axes, 
        latent_actions: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray, c: np.ndarray,
        action_scale: int):
    ax.cla() 
    
    try:
        prev_action, context = conn.recv()        
    except Exception as e:
        sys.exit()
    
    prev_action = prev_action.numpy()[0]
    contexts = context.expand(latent_actions.shape[0], context.shape[1])
    decoded_actions = _decode_latent_actions(
            decoder, latent_actions, contexts, x)
    u, v, w = np.split(decoded_actions, 3, axis=-1) 
    length = np.sqrt(u**2 + v**2 + w**2)
    scale = action_scale / 10
    u, v, w = u/length*scale, v/length*scale, w/length*scale
    ax.quiver(x, y, z, u, v, w)

    # Controller/latent current posistion.
    ax.plot(*prev_action, 'ro')
    # Small hacks to 'equalize' axes since matplotlib doesn't support it.
    ax.plot(0, 0, action_scale // 2, alpha=0)
    ax.plot(0, 0, -action_scale // 2, alpha=0)


def _decode_latent_actions(decoder, latent_actions, contexts, x):
    decoded_actions = decoder(latent=latent_actions, context=contexts)
    decoded_actions = decoded_actions.detach().numpy()[:, :3]
    decoded_actions = decoded_actions.reshape((x.shape[0], x.shape[1], 3))
    return decoded_actions
