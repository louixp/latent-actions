from argparse import ArgumentParser
import pickle
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class RealCenterOutDemonstrationDataset(Dataset):

    def __init__(self, 
            data_path: str,
            **kwargs):
        
        with open(data_path, "rb") as fp:
            episodes = pickle.load(fp)

        self.actions_joints = torch.tensor([
                step for episode in episodes['j_vel'] for step in episode],
                dtype=torch.float)
        self.joint_angles = torch.tensor([
                step for episode in episodes['j_pos'] for step in episode],
                dtype=torch.float)

        assert(len(self.actions_joints) == len(self.joint_angles))
        
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
        return parser
