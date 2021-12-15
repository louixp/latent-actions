"""Latent Action Conditional AutoEncoder."""

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything


class VAE(LightningModule):

    def __init__(self, 
            context_dim: int = 19,
            action_dim: int = 4,
            latent_dim: int = 2,
            enc_dims: Tuple[int] = (3, 2),
            dec_dims: Tuple[int] = (3, ),
            lr: float = 1e-2, 
            kl_coeff: float = 0.1, 
            p_dropout: float = 0.5):
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.p_dropout = p_dropout

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


class ConditionalVAE(VAE):

    def __init__(self, 
            context_dim: int = 19,
            action_dim: int = 4,
            latent_dim: int = 2,
            enc_dims: Tuple[int] = (12, 4),
            dec_dims: Tuple[int] = (12, ),
            lr: float = 1e-2, 
            kl_coeff: float = 0.1, 
            p_dropout: float = 0.5):
        super().__init__()

        self.save_hyperparameters()
        
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.p_dropout = p_dropout

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
        return self.decoder(latent)
    
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
    from dataset import DemonstrationDataset
    dataset = DemonstrationDataset.from_baselines_rl_zoo(
            "../../rl-baselines3-zoo/demonstration.pkl")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ConditionalVAE()
    trainer = Trainer()
    trainer.fit(model, dataloader)
