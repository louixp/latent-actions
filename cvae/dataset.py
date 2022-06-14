from argparse import ArgumentParser
import pickle
from typing import Iterable, Tuple
import random
import torch
from torch.utils.data import Dataset

class DemonstrationDataset(Dataset):

    def __init__(self,
            path: str, *,
            include_goal: bool,
            include_joint_angles: bool,
            dof: int,
            keep_success: bool,
            size_limit: int = None,
            add_adversarial_pos: bool):
        print(f"Initializing DemonstrationDataset with arguements: {locals()}")

        with open(path, "rb") as fp:
            data = pickle.load(fp)
            episodes = data["episodes"]
            rewards = data["rewards"]
            successes = data["successes"]

        if keep_success:
            episodes = [epi for epi, suc in zip(episodes, successes) if suc]

        # FIXME: Using position to identify observations and actions in a step
        # is a horrible idea.
        observations = [
                torch.squeeze(torch.from_numpy(step[0]["observation"]))
                for epi in episodes for step in epi]

        # Remove time wrapper feature.
        self.contexts = [o[:-1] for o in observations]

        if include_goal:
            goals = [
                    torch.squeeze(torch.from_numpy(step[0]["desired_goal"]))
                    for epi in episodes for step in epi]
            self.contexts = [torch.cat(c) for c in zip(self.contexts, goals)]

        if include_joint_angles:
            joint_angles = [
                    torch.squeeze(torch.tensor(step[2]))
                    for epi in episodes for step in epi]
            self.contexts = [
                    torch.cat(c) for c in zip(self.contexts, joint_angles)]

        
        assert dof in {3, 7}, f"DoF {dof} not supported"
        if dof == 3:
            self.actions = [torch.squeeze(torch.from_numpy(step[1]))
                    for epi in episodes for step in epi]
        elif dof == 7:
            self.actions = [
                    torch.cat((
                        torch.squeeze(torch.from_numpy(step[4])),
                        torch.squeeze(torch.from_numpy(step[1]))[3:])
                        ).to(torch.float32)
                    for epi in episodes for step in epi]

        if size_limit is not None:
            self.contexts = self.contexts[:size_limit]
            self.actions = self.actions[:size_limit]

        if add_adversarial_pos:
            # randomly redistribute object position
            newContexts = []

            random.seed(2)
            for c in self.contexts:
                x,y,z = c[7:10].tolist()# obj pos
                xx=random.random()*2-1 # fake pos
                yy=random.random()*2-1
                zz=random.random()*2-1

                if random.random() > 0.5:
                    c[8],c[9],c[10] = x,y,z
                    newvec = torch.cat((c,torch.Tensor([xx,yy,zz])))
                else:
                    c[8],c[9],c[10] = xx,yy,zz
                    newvec = torch.cat((c,torch.Tensor([x,y,z])))

                newContexts.append(newvec)

            self.context = newContexts
             #NOT DONE SJ

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[idx], self.actions[idx]

    def get_context_dim(self) -> int:
        return self.contexts[0].shape[0]

    def get_action_dim(self) -> int:
        return self.actions[0].shape[0]

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--include_goal", action="store_true")
        parser.add_argument("--include_joint_angles", action="store_true")
        parser.add_argument("--dof", type=int, default=3, choices=(3, 7))
        parser.add_argument("--keep_success", action="store_true")
        parser.add_argument("--size_limit", type=int)
        parser.add_argument("--add_adversarial_pos", action="store_true")
        return parser
