from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
import wandb

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
		**kwargs)

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

        #wandb variable must be passed in to set or unset
        self.no_wandb = kwargs["no_wandb"]

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

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        context = batch[0]
        distance = self._batch_distance_to_object(context)
        divergence = self._decoder_divergence(context)
        return distance, divergence

    def validation_epoch_end(self, val_outs):
        distance, divergence = zip(*val_outs)
        distance, divergence = torch.cat(distance), torch.cat(divergence)
        data = [[x, y] for x, y in zip(distance, divergence)]

        if not self.no_wandb:
            table = wandb.Table(
                    data=data, columns=["object distance", "divergence"])
            wandb.log({
                f"decoder divergence epoch {self.current_epoch}":
                wandb.plot.scatter(table, "object distance", "divergence")})


        print("MODEL WEIGHTS")
        # for param in self.parameters(): # works well
        #     print(param.data)
        # print weights of decoder layer for

        # decoder weights
        # print("decoder weights")
        decoderWeights = self.decoder[0].weight
        # print(decoderWeights.shape) #10x21 (input is 21 and output is 10) latent is 2 context is 19

        # measure magnitude of latent
        latentpart = decoderWeights[:,:2] #10x2
        print("latent part of decoder:", latentpart.shape, torch.norm(latentpart))
        # print(latentpart.shape)
        # print(latentpart)
        # print(torch.norm(latentpart))

        # measure magnitude of context
        contextpart = decoderWeights[:,2:] #10x19
        print("context part of decoder:", contextpart.shape, torch.norm(contextpart))
        # print(contextpart.shape)
        # print(contextpart)
        # print(torch.norm(contextpart))


    def fixed_point_constraint(self, context, z):
        zero = torch.zeros_like(z)
        x_dec = torch.cat([zero, context], dim = 1)
        action_zero = self.decoder(x_dec)
        return torch.linalg.norm(action_zero)

    def _batch_jacobian(self, x):
        """Computes the Jacobian at a batch of inputs x."""
        # Adapted from https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/5
        def _func_sum(x):
            return self.decoder(x).sum(dim=0)
        return torch.autograd.functional.jacobian(
                _func_sum, x, create_graph=True).permute(1,0,2)

    def _decoder_divergence(self, context):
        if self.latent_dim != 2:
            print("WARNING: latent dim != 2 and decoder divergence may not be meaningful.")
        zero = torch.zeros(context.shape[0], self.latent_dim)
        x_dec = torch.cat([zero, context], dim = 1)
        jacobian = self._batch_jacobian(x_dec)
        return jacobian[:, 0, 0] + jacobian[:, 1, 1]

    def _batch_distance_to_object(self, context):
        ee_pos = context[:, 0:3]
        obj_pos = context[:, 7:10]
        return torch.norm(ee_pos - obj_pos, dim=-1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(ConditionalVAE, ConditionalVAE).add_model_specific_args(
                parent_parser)
        parser.add_argument("--fixed_point_coeff", type=float, default=0)
        parser.add_argument("--dropout", type=float, default=0)
        return parser
