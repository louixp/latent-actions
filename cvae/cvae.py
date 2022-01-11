"""Latent Action Conditional AutoEncoder."""

from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary


class VAE(LightningModule):

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
            **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.kl_schedule = kl_schedule 
        
        if activation == "tanh":
            self.Activation = nn.Tanh
        else:
            self.Activation = nn.ReLU
        
        enc_dims = [action_dim] + list(enc_dims)
        enc_layers = [
                layer for d_in, d_out in zip(enc_dims[:-1], enc_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        enc_layers.pop()
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + list(dec_dims) + [action_dim]
        dec_layers = [
                layer for d_in, d_out in zip(dec_dims[:-1], dec_dims[1:])
                for layer in [nn.Linear(d_in, d_out), self.Activation()]]
        dec_layers.pop()
        self.decoder = nn.Sequential(*dec_layers)

        self.fc_mu = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_var = nn.Linear(enc_dims[-1], latent_dim)

    def set_kl_scheduler(self, n_steps: int):
        import kl_scheduler
        if self.kl_schedule == "monotonic":
            print("Using monotonic KL annealing schedule.")
            self.kl_scheduler = kl_scheduler.monotonic_annealing_scheduler(
                    n_steps)
        elif self.kl_schedule == "cyclical":
            print("Using cyclical KL annealing schedule.")
            self.kl_scheduler = kl_scheduler.cyclical_annealing_scheduler(
                    n_steps)
        else:
            assert isinstance(self.kl_coeff, float)
            print("Using constant KL schedule.")
            self.kl_scheduler = kl_scheduler.constant_scheduler(self.kl_coeff)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        return self.decoder(latent)
    
    def step(self, batch, batch_idx, kl_coeff):
        _, action = batch
        x = self.encoder(action)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        action_recon = self.decoder(z)
        return self.compute_vae_loss(action, action_recon, p, q, kl_coeff)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def compute_vae_loss(self, action, action_recon, p, q, kl_coeff):
        recon_loss = F.mse_loss(action_recon, action, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, next(self.kl_scheduler))
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, 
            on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, self.kl_coeff)
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

        parser.add_argument(
                "--kl_schedule", type=str, default="cyclical", 
                choices=["monotonic", "cyclical"], 
                help="KL schedule.") 
        parser.add_argument(
                "--kl_coeff", type=float, default=1, 
                help="KL coeff for constant schedule.")

        parser.add_argument("--latent_dim", type=int, default=2)
        parser.add_argument("--enc_dims", nargs='+', type=int, default=(3, 2))
        parser.add_argument("--dec_dims", nargs='+', type=int, default=(3, ))
        return parser


class ConditionalVAE(VAE):

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
                action_dim=action_dim)
        
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
        return parser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
            "--model_class", default="cVAE", type=str, choices=["VAE", "cVAE"])
    script_args, _ = parser.parse_known_args()
    if script_args.model_class == "VAE":
        ModelClass = VAE
    else:
        ModelClass = ConditionalVAE

    parser.add_argument("--batch_size", type=int, default=32)
    parser = ModelClass.add_model_specific_args(parser)
    args = parser.parse_args()

    from dataset import DemonstrationDataset
    dataset = DemonstrationDataset.from_baselines_rl_zoo(
            "../../rl-baselines3-zoo/demonstration.pkl")
    train_set, test_set = torch.utils.data.random_split(
            dataset, [int(len(dataset) * .8), int(len(dataset) * .2)])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
   
    wandb_logger = WandbLogger(project="latent-action", entity="ucla-ncel-robotics")
    trainer = Trainer(logger=wandb_logger)
        
    model = ModelClass(**vars(args))
    model.set_kl_scheduler(n_steps=trainer.max_epochs*len(train_loader)) 
    
    print(model)
    model_summary = ModelSummary(model)
    wandb_logger.log_hyperparams({
        "total_parameters": model_summary.total_parameters,
        "trainable_parameters": model_summary.trainable_parameters})
    
    trainer.fit(model, train_loader, test_loader)
