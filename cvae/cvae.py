from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
import wandb
import pytorch_lightning

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

        # weight weightRegularization
        # loss += self.decoderWeightRegularization()
        self.gradientMagnitdueRegularization(x_dec,logs) # is only logging, not regularizing

        return loss, logs

    def decoderWeightRegularization(self):
        # add weight regularization term
        decoderWeights = self.decoder[0].weight
        latentpart = decoderWeights[:,:2] #10x2 latent
        contextpart = decoderWeights[:,2:] #10x19 context
        # average norm: average of square each value. better comparison then L2 norm, since latent numel is smaller
        # avglatentnorm = torch.sum(latentpart*latentpart) / torch.numel(latentpart)
        # avgcontextnorm = torch.sum(contextpart*contextpart) / torch.numel(contextpart)
        avglatentnorm = torch.max(latentpart*latentpart)
        avgcontextnorm = torch.max(contextpart*contextpart)
        return avgcontextnorm * 10

    def gradientMagnitdueRegularization(self,x_dec,logs):

        # regularize by introducing action norm gradient w.r.t context
        contextOnlyInput = x_dec.clone(); contextOnlyInput[:,:2] = 0
        contextContribution = self._action_norm_gradient(contextOnlyInput)[:,:,2:]
        # averageContextContributionMagnitude = torch.norm(contextContribution)
        averageContextContributionMagnitude = torch.sum(contextContribution*contextContribution) / (torch.numel(contextContribution)/contextContribution.shape[0])
        #^normally it's shape is 32,32,19, but only 32,19 values are propagated. shows how those 19 values influence 32 batch sized outputs. so for individual influence, it isi average over 19*32 values

        # regularize by introducing action norm gradient w.r.t z
        zOnlyInput = x_dec.clone(); zOnlyInput[:,2:] = 0
        zContribution = self._action_norm_gradient(zOnlyInput)[:,:,:2]
        # averageZContributionMagnitude = torch.norm(zContribution)
        averageZContributionMagnitude = torch.sum(zContribution*zContribution) / (torch.numel(zContribution)/contextContribution.shape[0])
        #^normally it's shape is 32,32,2. but only 32,2 values are propagated.shows how those 19 values influence 32 batch sized outputs. so for individual influence, it isi average over 2*32 values

        if not isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            print("context contribution",averageContextContributionMagnitude)
            print("z contribution",averageZContributionMagnitude)
        if isinstance(self.logger, pytorch_lightning.loggers.WandbLogger): # log result
            logs["average_context_contribution_to_magnitude_of_output"] = averageContextContributionMagnitude
            logs["average_z_contribution_to_magnitude_of_output"] = averageZContributionMagnitude

        # return (averageContextContributionMagnitude-averageZContributionMagnitude) * 1000 #weight I decided
        return 0


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

        if isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            table = wandb.Table(
                    data=data, columns=["object distance", "divergence"])
            wandb.log({
                f"decoder divergence epoch {self.current_epoch}":
                wandb.plot.scatter(table, "object distance", "divergence")})

        # log decoder first layer's weight
        decoderWeights = self.decoder[0].weight
        latentpart = decoderWeights[:,:2] #10x2 latent
        contextpart = decoderWeights[:,2:] #10x19 context
        # average norm: average of square each value. better comparison then L2 norm, since latent numel is smaller
        avglatentnorm = torch.sum(latentpart*latentpart) / torch.numel(latentpart)
        avgcontextnorm = torch.sum(contextpart*contextpart) / torch.numel(contextpart)
        maxlatentnorm = torch.max(latentpart*latentpart)
        maxcontextnorm = torch.max(contextpart*contextpart)
        if isinstance(self.logger, pytorch_lightning.loggers.WandbLogger): # log result
            wandb.log({"first_layer_latent_magnitude": avglatentnorm,
                    "first_layer_context_magnitude": avgcontextnorm,
                    "first_layer_max_latent_magnitude": maxlatentnorm,
                    "first_layer_max_context_magnitude": maxcontextnorm,})
        else: # print result
            print("first layer latent magnitude:", latentpart.shape, avglatentnorm)
            print("first layer context magnitude:", contextpart.shape, avgcontextnorm)


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

    def _action_norm_gradient(self,x):
        def _func(x):
            return torch.norm(self.decoder(x),dim=1)
        return torch.autograd.functional.jacobian(_func,x)

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
