from argparse import ArgumentParser
from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Set

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Step:
    joint_velocity: np.array
    ee_velocity: np.array
    context: Dict 
    gripper_velocity: np.array = None
    human_action: np.array = np.array([]) 

@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)

    def append(self, step: Step):
        if not isinstance(step, Step):
            raise ValueError("Each element of an Episode must be a Step.")
        self.steps.append(step)

@dataclass
class EpisodicDataset:
    """EpisodicDataset defines a generic data structure that encompasses all 
    data collected on real or simulated robot, across all tasks.
    """
    episodes: List[Episode] = field(default_factory=list) 

    def append(self, episode: Episode):
        if not isinstance(episode, Episode):
            raise ValueError(
                    "Each element of an Episodic Dataset must be an Episode.")
        self.episodes.append(episode)

    def dump(self, data_path: str):
        with open(data_path, "wb") as fp:
            pickle.dump(self, fp)
    
    @staticmethod
    def load(data_path: str):
        with open(data_path, "rb") as fp:
            return pickle.load(fp)


class DemonstrationDataset(Dataset):
    """DemonstrationDataset is a torch Dataset that performs the appropiate
    transformations required for VAE training."""

    def __init__(self, 
            episodic_dataset: EpisodicDataset, 
            action_space: str, 
            size_limit: int,
            **kwargs):
        
        excluded_context_keys = set(
                local_var_key[len("exclude_context_feature_"):] 
                for local_var_key in locals() 
                if local_var_key.startswith("exclude_context_feature_"))
        self.contexts = np.array([
                [ctx_val for ctx_key, ctx_val in contexts.items() 
                    if ctx_key not in excluded_context_keys] 
                for episode in episodic_dataset.episodes 
                for step in episode.episode])
        
        assert(action_space in ["ee", "joints"])
        if action_space == "ee":
            self.actions = np.array([
                np.concatenate(step.ee_velocity, step.gripper_velocity)
                for episode in episodic_dataset.episodes 
                for step in episode.episode])
        else:
            self.actions = np.array([
                np.concatenate([step.joint_velocity, step.gripper_velocity])
                for episode in episodic_dataset.episodes 
                for step in episode.episode])

        self.human_actions = np.array([step.human_action
            for episode in episodic_dataset.episodes 
            for step in episode.episode])

        assert(len(self.contexts) == len(self.actions) == len(self.human_actions))

        if size_limit is not None:
            self.contexts = self.contexts[:size_limit]
            self.actions = self.actions[:size_limit]
            self.human_actions = self.actions[:size_limit]

    def __len__(self) -> int:
        return len(self.contexts)
       
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
                "context": self.contexts[idx],
                "action": self.actions[idx],
                "human_action": self.human_actions[idx]}

    def get_context_dim(self) -> int:
        return self.contexts[0].shape[0]

    def get_action_dim(self) -> int:
        return self.actions[0].shape[0]

    def get_human_action_dim(self) -> int:
        return self.human_actions[0].shape[0]
        
    @staticmethod
    def add_dataset_specific_args(
            parent_parser: ArgumentParser, 
            episodic_dataset: EpisodicDataset):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        context_keys = episodic_dataset.episodes[0].steps[0].context.keys() 
        for context_key in context_keys:
            parser.add_argument(
                    f"--exclude_context_feature_{context_key}", 
                    action="store_true") 
        parser.add_argument(
                "--action_space", type=str, choices=["ee", "joints"], 
                required=True)
        parser.add_argument("--size_limit", type=int)
        return parser
