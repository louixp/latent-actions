""" Latent Action Conditional AutoEncoder."""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F

class ConditionalVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_mu = nn.Linear(4, 2)
        self.fc_var = nn.Linear(4, 2)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> List[torch.Tensor]:
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z, context)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([action, context], dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(22, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, context], dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
