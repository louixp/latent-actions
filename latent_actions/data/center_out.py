from argparse import ArgumentParser
import pickle
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class CenterOutDemonstrationDataset(Dataset):

    def __init__(self, 
            data_path: str, *, 
            radius_cutoff: float = np.inf,
            size_limit: int = None,
            subtract_neutral_from_context: bool = False,
            **kwargs):
        print(f"Initializing CenterOutDemonstrationDataset with: {locals()}")
        
        with open(data_path, "rb") as fp:
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
        
        if subtract_neutral_from_context:
            self.joint_angles -= torch.tensor(
                    [0., 0.41, 0., -1.85, 0., 2.26, 0.79])
        
        assert(len(self.actions_joints) == len(self.actions_ee) == len(self.joint_angles))

        if size_limit is not None:
            self.actions_joints = self.actions_joints[:size_limit]
            self.actions_ee = self.actions_ee[:size_limit]
        
    def __len__(self) -> int:
        return len(self.actions_joints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.joint_angles[idx], self.actions_joints[idx] 
    
    def get_context_dim(self) -> int:
        return self.joint_angles[0].shape[0]

    def get_action_dim(self) -> int:
        return self.actions_joints[0].shape[0]

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--radius_cutoff", type=float, default=np.inf)
        parser.add_argument("--size_limit", type=int)
        parser.add_argument("--subtract_neutral_from_context", action="store_true")
        return parser
