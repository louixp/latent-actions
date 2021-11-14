import pickle

import torch
from torch.utils.data import Dataset

class DemonstrationDataset(Dataset):

    def __init__(self, observations, actions):
        assert len(observations) == len(actions), \
            "Different amount of observations and actions"
        self.observations = observations
        self.actions = actions

    @classmethod
    def from_baselines_rl_zoo(cls, path: str):
        """Constructs dataset from episodes generated by RL agents from 
            stable-baselines3. 
        """
        with open(path, "rb") as fp:
            trajectories = pickle.load(fp)
        obs, actions = zip(*trajectories)
        obs = [torch.from_numpy(o["observation"]) for o in obs]
        actions = [torch.from_numpy(a) for a in actions]
        return DemonstrationDataset(obs, actions)
        
    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx] 
