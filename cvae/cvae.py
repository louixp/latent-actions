from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
import pytorch_lightning
import wandb

from . import vae
from . import distance_gated_decoder


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
            div_coeff: float = 0, 
            div_clip: float = float("inf"),
            distance_gated_decode: bool = False,
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
        self.fixed_point_coeff = fixed_point_coeff
        self.div_coeff = div_coeff
        self.div_clip = div_clip
        self.distance_gated_decode = distance_gated_decode
        
        enc_dims = [action_dim + context_dim] + list(enc_dims)
        enc_layers = [
                layer for d_in, d_out in zip(enc_dims[:-1], enc_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        enc_layers.pop()
        self.encoder = nn.Sequential(*enc_layers)
        
        if distance_gated_decode:
            self.decoder = distance_gated_decoder.DistanceGatedDecoder(
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    action_dim=action_dim,
                    dec_dims=dec_dims,
                    activation=activation,
                    **kwargs)
        else:
            self.decoder = distance_gated_decoder.MLPDecoder(
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    action_dim=action_dim,
                    dec_dims=dec_dims,
                    activation=activation)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        return self.decoder(latent=latent, context=context)
    
    def step(self, batch, batch_idx, kl_coeff):
        context, action = batch
        context = self.dropout(context)
        
        x = torch.cat([action, context], dim = 1)
        x = self.encoder(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        
        action_recon = self.decoder(latent=z, context=context)
        
        loss, logs = self.compute_vae_loss(action, action_recon, p, q, kl_coeff)
        fixed_point_loss = self.fixed_point_constraint(context, z)
        logs["fixed_point_loss"] = fixed_point_loss
        loss += self.fixed_point_coeff * fixed_point_loss
        
        clipped_divergence = torch.minimum(
                self._decoder_divergence(context), 
                torch.tensor([self.div_clip]))
        loss -= self.div_coeff * clipped_divergence.mean()
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
        self.log("max_scaled_mean_divergence", 
                divergence.mean() / torch.max(divergence))

        if isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            table = wandb.Table(
                    data=data, columns=["object distance", "divergence"])
            wandb.log({
                f"decoder divergence epoch {self.current_epoch}": 
                wandb.plot.scatter(table, "object distance", "divergence")})

    def fixed_point_constraint(self, context, z):
        zero = torch.zeros_like(z)
        action_zero = self.decoder(latent=zero, context=context)
        return torch.linalg.norm(action_zero)

    def _batch_jacobian(self, context):
        """Computes the Jacobian at a batch of inputs x."""
        # Adapted from https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/5
        def _func_sum(x):
            zero = torch.zeros(context.shape[0], self.latent_dim)
            return self.decoder(latent=zero, context=x).sum(dim=0)
        return torch.autograd.functional.jacobian(
                _func_sum, context, create_graph=True).permute(1,0,2)

    def _decoder_divergence(self, context):
        if self.latent_dim != 2:
            print("WARNING: latent dim != 2 and decoder divergence may not be meaningful.")
        jacobian = self._batch_jacobian(context) 
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
        parser.add_argument("--div_coeff", type=float, default=0)
        parser.add_argument("--div_clip", type=float, default=float("inf"))
        parser.add_argument("--distance_gated_decode", action="store_true")
        parser.add_argument("--gated_feature_dim", type=int)
        parser.add_argument("--context_featurizer_dims", nargs='+', type=int)
        parser.add_argument("--latent_featurizer_dims", nargs='+', type=int)
        return parser
