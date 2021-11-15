from time import sleep

import numpy as np
import torch
import gym
import panda_gym

from models.cvae import ConditionalVAE

env = gym.make('PandaPickAndPlace-v1', render=True)
cvae = ConditionalVAE()
latent_action_decoder = cvae.decoder

obs = env.reset()
done = False
while not done:
    latent_action = torch.rand((1, 2))
    context = torch.from_numpy(obs['observation'])
    context = torch.unsqueeze(context, 0).float()
    
    action = latent_action_decoder(latent_action, context)
    action = action.detach().numpy()
    action = np.squeeze(action)

    obs, reward, done, info = env.step(action)
    sleep(0.01)

env.close()
