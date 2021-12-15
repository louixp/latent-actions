"""Latent Action Conditional AutoEncoder."""

from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


class VAE(LightningModule):

    def __init__(self, 
            context_dim: int = 19,
            action_dim: int = 4,
            latent_dim: int = 2,
            enc_dims: Tuple[int] = (3, 2),
            dec_dims: Tuple[int] = (3, ),
            lr: float = 1e-4,
            kl_coeff: float = 0.1,
            **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = lr
        self.kl_coeff = kl_coeff

        self.encoder = nn.Sequential(
                nn.Linear(action_dim, enc_dims[0]),
                nn.ReLU(),
                nn.Linear(enc_dims[0], enc_dims[1]))
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, dec_dims[0]),
                nn.ReLU(),
                nn.Linear(dec_dims[0], action_dim))

        self.fc_mu = nn.Linear(enc_dims[1], latent_dim)
        self.fc_var = nn.Linear(enc_dims[1], latent_dim)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        return self.decoder(latent)
    
    def step(self, batch, batch_idx):
        _, action = batch
        x = self.encoder(action)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        action_recon = self.decoder(z)
        return self.compute_vae_loss(action, action_recon, p, q)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def compute_vae_loss(self, action, action_recon, p, q):
        recon_loss = F.mse_loss(action_recon, action, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=2)
        parser.add_argument("--enc_dims", nargs='+', type=int, default=(3, 2))
        parser.add_argument("--dec_dims", nargs='+', type=int, default=(3, ))
        return parser


class ConditionalVAE(VAE):

    def __init__(self, 
            context_dim: int = 19,
            action_dim: int = 4,
            latent_dim: int = 2,
            enc_dims: Tuple[int] = (12, 4),
            dec_dims: Tuple[int] = (12, ),
            lr: float = 1e-4, 
            kl_coeff: float = 0.1,
            **kwargs): 
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = lr
        self.kl_coeff = kl_coeff

        self.encoder = nn.Sequential(
                nn.Linear(action_dim + context_dim, enc_dims[0]),
                nn.ReLU(),
                nn.Linear(enc_dims[0], enc_dims[1]))
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim + context_dim, dec_dims[0]),
                nn.ReLU(),
                nn.Linear(dec_dims[0], action_dim))

        self.fc_mu = nn.Linear(enc_dims[1], latent_dim)
        self.fc_var = nn.Linear(enc_dims[1], latent_dim)

    def forward(self, *, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inference only. See step() for training."""
        z = torch.cat([latent, context], dim = 1)
        return self.decoder(z)
    
    def step(self, batch, batch_idx):
        context, action = batch
        
        x = torch.cat([action, context], dim = 1)
        x = self.encoder(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        
        z = torch.cat([z, context], dim = 1)
        action_recon = self.decoder(z)
        return self.compute_vae_loss(action, action_recon, p, q)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
            "--model_class", default="VAE", type=str, choices=["VAE", "cVAE"])
    script_args, _ = parser.parse_known_args()
    if script_args.model_class == "VAE":
        ModelClass = VAE 
    else:
        ModelClass = ConditionalVAE

    parser.add_argument("--batch_size", type=int, default=32)
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    from dataset import DemonstrationDataset
    dataset = DemonstrationDataset.from_baselines_rl_zoo(
            "../../rl-baselines3-zoo/demonstration.pkl")
    train_set, test_set = torch.utils.data.random_split(
            dataset, [int(len(dataset) * .8), int(len(dataset) * .2)])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    
    model = VAE(**vars(args))
    wandb_logger = WandbLogger(project="latent-action")
    trainer = Trainer(logger=wandb_logger)
    trainer.fit(model, train_loader, test_loader)
