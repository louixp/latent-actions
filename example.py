from time import sleep

import numpy as np
import torch
import gym
import panda_gym

from cvae.cvae import ConditionalVAE
from controller import Controller

controller = Controller(scale=10)

cvae = ConditionalVAE()
latent_action_decoder = cvae.decoder

env = gym.make('PandaPickAndPlace-v1', render=True).env
obs = env.reset()
done = False
while not done:
    latent_action = controller.get_action()
    context = torch.from_numpy(obs['observation'])
    context = torch.unsqueeze(context, 0).float()
    
    action = latent_action_decoder(latent_action, context)
    action = action.detach().numpy()
    action = np.squeeze(action)

    obs, reward, done, info = env.step(action)
    sleep(0.1)

env.close()
