from argparse import ArgumentParser
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class AlignedDecoder(LightningModule):

    def __init__(self,
            decoder: LightningModule,
            lr: float = 1e-4,
            align_dims: Tuple[int] = None, 
            activation: str = "relu",
            latent_dim: int = 2,
            **kwargs):
        
        super().__init__()

        self.save_hyperparameters("align_dims", "activation", "latent_dim")
        self.lr = lr
        self.latent_dim = latent_dim

        self.decoder = decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        if activation == "tanh":
            self.Activation = nn.Tanh
        else:
            self.Activation = nn.ReLU
        
        if align_dims is not None:
            dims = [latent_dim] + align_dims + [latent_dim]
            layers = [
                    layer for d_in, d_out in zip(dims[:-1], dims[1:])
                    for layer in [nn.Linear(d_in, d_out), self.Activation()]]
            layers.pop()
            self.aligner = nn.Sequential(*layers)
        else:
            self.aligner = lambda x: x

    def forward(self, *,
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        z = self.aligner(latent)
        return self.decoder(latent=z, context=context) 

    def step(self, batch, batch_idx):
        context, action = batch
        z = self.aligner(action[:, :self.latent_dim])
        action_recon = self.decoder(latent=z, context=context)
        
        loss = F.mse_loss(action_recon, action, reduction="mean")
        logs = {
                "align_loss": loss,
                "manifold_distance": self.mean_pointwise_manifold_distance(
                    context)}
        return loss, logs

    def mean_pointwise_manifold_distance(
            self, context: torch.Tensor) -> torch.Tensor:
        z = torch.rand((context.shape[0], self.latent_dim))
        z_aligned = self.aligner(z)
        action_aligned = self.decoder(latent=z_aligned, context=context)
        action_unaligned = self.decoder(latent=z, context=context)
        dist = F.mse_loss(action_aligned, action_unaligned, reduction="mean")
        return dist 

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
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument(
                "--activation", type=str, default="relu", 
                choices=["tanh", "relu"])
        parser.add_argument("--align_dims", nargs='+', type=int)
        return parser

