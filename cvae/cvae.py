from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn

from . import vae


class ConditionalVAE(vae.VAE):

    def __init__(self, 
            latent_dim: int,
            enc_dims: Tuple[int],
            dec_dims: Tuple[int],
            lr: float, 
            kl_coeff: float,
            kl_schedule: str = "constant",
            activation: str = "relu",
            context_dim: int = 19,
            action_dim: int = 4,
            fixed_point_coeff: int = 0,
            dropout: float = 0,
            **kwargs): 
        super().__init__(
                latent_dim=latent_dim, 
                enc_dims=enc_dims, 
                dec_dims=dec_dims,
                lr=lr,
                kl_coeff=kl_coeff,
                kl_schedule=kl_schedule,
                activation=activation,
                context_dim=context_dim,
                action_dim=action_dim,
                include_joint_angles=include_joint_angles)

        self.dropout = nn.Dropout(dropout)
        
        enc_dims = [action_dim + context_dim] + list(enc_dims)
        enc_layers = [
                layer for d_in, d_out in zip(enc_dims[:-1], enc_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        enc_layers.pop()
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim + context_dim] + list(dec_dims) + [action_dim]
        dec_layers = [
                layer for d_in, d_out in zip(dec_dims[:-1], dec_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        dec_layers.pop()
        self.decoder = nn.Sequential(*dec_layers)

        self.fixed_point_coeff = fixed_point_coeff

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        z = torch.cat([latent, context], dim = 1)
        return self.decoder(z)
    
    def step(self, batch, batch_idx, kl_coeff):
        context, action = batch
        context = self.dropout(context)
        
        x = torch.cat([action, context], dim = 1)
        x = self.encoder(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        
        x_dec = torch.cat([z, context], dim = 1)
        action_recon = self.decoder(x_dec)
        
        loss, logs = self.compute_vae_loss(action, action_recon, p, q, kl_coeff)
        fixed_point_loss = self.fixed_point_constraint(context, z)
        logs["fixed_point_loss"] = fixed_point_loss
        loss += self.fixed_point_coeff * fixed_point_loss
        return loss, logs

    def fixed_point_constraint(self, context, z):
        zero = torch.zeros_like(z)
        x_dec = torch.cat([zero, context], dim = 1)
        action_zero = self.decoder(x_dec)
        return torch.linalg.norm(action_zero)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(ConditionalVAE, ConditionalVAE).add_model_specific_args(
                parent_parser)
        parser.add_argument("--fixed_point_coeff", type=float, default=0)
        parser.add_argument("--dropout", type=float, default=0)
        return parser
