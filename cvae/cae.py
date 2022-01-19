from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import cvae


class ConditionalAE(cvae.ConditionalVAE):

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
                kl_coeff=0,
                kl_schedule="none",
                activation=activation,
                context_dim=context_dim,
                action_dim=action_dim)

        enc_dims = [action_dim + context_dim] + list(enc_dims) + [latent_dim]
        enc_layers = [
                layer for d_in, d_out in zip(enc_dims[:-1], enc_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        enc_layers.pop()
        self.encoder = nn.Sequential(*enc_layers)

        del self.fc_mu
        del self.fc_var
    
    def step(self, batch, batch_idx):
        context, action = batch
        context = self.dropout(context)
        
        x = torch.cat([action, context], dim = 1)
        z = self.encoder(x)
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

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True,
            on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

