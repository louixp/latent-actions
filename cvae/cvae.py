"""Latent Action Conditional AutoEncoder."""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything


class ConditionalVAE(LightningModule):

    def __init__(self, 
            lr: float = 1e-2, 
            kl_coeff: float = 0.1, 
            p_dropout: float = 0.5):
        super().__init__()
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.p_dropout = p_dropout

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_mu = nn.Linear(4, 2)
        self.fc_var = nn.Linear(4, 2)

    def forward(self, 
            action: torch.Tensor, 
            context: torch.Tensor, 
            return_dist: bool = False) -> List[torch.Tensor]:
        masked_context = F.dropout(
                context, p=self.p_dropout, training=self.training) 
        x = self.encoder(action, masked_context)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        if return_dist:
            return self.decoder(z, masked_context), p, q
        else:
            return self.decoder(z, masked_context)
    
    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        context, action = batch
        action_hat, p, q = self.forward(action, context, return_dist=True)

        recon_loss = F.mse_loss(action_hat, action, reduction="mean")

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


class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, 
            action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([action, context], dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, 
            latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, context], dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    from dataset import DemonstrationDataset
    dataset = DemonstrationDataset.from_baselines_rl_zoo(
            "../../rl-baselines3-zoo/demonstration.pkl")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ConditionalVAE()
    trainer = Trainer()
    trainer.fit(model, dataloader)
