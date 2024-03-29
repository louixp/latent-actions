from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import cae


class GaussianBC(cae.ConditionalAE):

    def __init__(self, 
            latent_dim: int,
            dec_dims: Tuple[int],
            lr: float, 
            activation: str = "relu",
            context_dim: int = 19,
            action_dim: int = 4,
            fixed_point_coeff: int = 0,
            dropout: float = 0,
            **kwargs): 
        super().__init__(
                latent_dim=latent_dim, 
                enc_dims=(32, ), 
                dec_dims=dec_dims,
                lr=lr,
                kl_coeff=0,
                kl_schedule="none",
                activation=activation,
                context_dim=context_dim,
                action_dim=action_dim)
        
        del self.encoder
    
    def step(self, batch, batch_idx):
        context, action = batch
        context = self.dropout(context)
        normal_dist = torch.distributions.Normal(
                torch.zeros((context.shape[0], self.latent_dim)), 
                torch.ones((context.shape[0], self.latent_dim)))
        
        z = normal_dist.sample()
        x_dec = torch.cat([z, context], dim = 1)
        action_recon = self.decoder(x_dec)
        
        recon_loss = F.mse_loss(action_recon, action, reduction="mean")
        fixed_point_loss = self.fixed_point_constraint(context, z)
        loss = recon_loss + self.fixed_point_coeff * fixed_point_loss
        logs = {
                "recon_loss": recon_loss,
                "fixed_point_loss": fixed_point_loss,
                "loss": loss}
        return loss, logs

