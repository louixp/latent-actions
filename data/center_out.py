import pickle
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class CenterOutDemonstrationDataset(Dataset):

    def __init__(self, 
            path: str, *, 
            radius_cutoff: float = np.inf,
            size_limit: int = None):
        
        with open(path, "rb") as fp:
            episodes = pickle.load(fp)

        self.actions_joints = torch.tensor([
                step['action_joints']
                for episode in episodes for step in episode 
                if step['radius'] < radius_cutoff], dtype=torch.float)
        self.actions_ee = torch.tensor([
                step['action_ee']
                for episode in episodes for step in episode 
                if step['radius'] < radius_cutoff], dtype=torch.float)
        self.joint_angles = torch.tensor([
                step['previous_joint_angles']
                for episode in episodes for step in episode 
                if step['radius'] < radius_cutoff], dtype=torch.float)

        assert(len(self.actions_joints) == len(self.actions_ee) == len(self.joint_angles))

        if size_limit is not None:
            self.actions_joints = self.actions_joints[:size_limit]
            self.actions_ee = self.actions_ee[:size_limit]
        
    def __len__(self) -> int:
        return len(self.actions_joints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.joint_angles[idx], self.actions_joints[idx] 

