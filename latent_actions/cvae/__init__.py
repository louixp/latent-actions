from latent_actions.cvae.vae import VAE
from latent_actions.cvae.cvae import ConditionalVAE 
from latent_actions.cvae.cae import ConditionalAE 
from latent_actions.cvae.gbc import GaussianBC

DECODER_CLASS = {
        "VAE": VAE,
        "cVAE": ConditionalVAE,
        "cAE": ConditionalAE,
        "gBC": GaussianBC}
