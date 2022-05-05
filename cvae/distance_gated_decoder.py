from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning
from pytorch_lightning import LightningModule


def make_mlp(
        dim_in: int, dims_hidden: Tuple[int], dim_out: int, activation: "str"
        ) -> nn.Module:
    if activation == "tanh":
        Activation = nn.Tanh
    else:
        Activation = nn.ReLU
    
    dims = [dim_in] + list(dims_hidden) + [dim_out]
    layers = [
            layer for d_in, d_out in zip(dims[:-1], dims[1:])
            for layer in [nn.Linear(d_in, d_out), Activation()]]
    layers.pop()
    return nn.Sequential(*layers)


def get_dist_ee_to_obj(context):
    ee_pos = context[:, 0:3]
    obj_pos = context[:, 7:10]
    return torch.norm(ee_pos - obj_pos, dim=-1)

def inverse_gating(*, context_features, latent_features, distance, alpha=10):
    # Since the initial distance is around 0.2, alpha is set to 10 such that
    # latent contributes more (~0.7) at initial position. 
    p = 1 / (1 + alpha * distance)
    p = p[:, None]
    return context_features * p + latent_features * (1-p)

class MLPDecoder(nn.Module):
    def __init__(self, 
            latent_dim: int,
            context_dim: int,
            action_dim: int, 
            dec_dims: Tuple[int],
            activation: str, **kwargs): 
        super().__init__()

        self.decoder = make_mlp(
            latent_dim+context_dim, dec_dims, action_dim, activation)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, context], dim = 1)
        return self.decoder(x)


class DistanceGatedDecoder(nn.Module):
    def __init__(self, 
            latent_dim: int,
            gated_feature_dim: int,
            context_dim: int,
            action_dim: int, 
            context_featurizer_dims: Tuple[int],
            latent_featurizer_dims: Tuple[int],
            dec_dims: Tuple[int],
            activation: str, **kwargs): 
        super().__init__()
        
        self.context_featurizer = make_mlp(
            context_dim, context_featurizer_dims, gated_feature_dim, activation)
        self.latent_featurizer = make_mlp(
            latent_dim, latent_featurizer_dims, gated_feature_dim, activation)
        self.decoder = make_mlp(
            gated_feature_dim, dec_dims, action_dim, activation)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_features = self.context_featurizer(context)
        latent_features = self.latent_featurizer(latent)
        distance = get_dist_ee_to_obj(context)
        gated_features = inverse_gating(
                context_features=context_features, 
                latent_features=latent_features, 
                distance=distance) 
        return self.decoder(gated_features)
