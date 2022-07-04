from argparse import ArgumentParser
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from cvae.vae import VAE
from data.center_out import CenterOutDemonstrationDataset


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--radius_cutoff", type=float, default=np.inf)
parser.add_argument("--size_limit", type=int)
parser.add_argument("--no_wandb", action="store_true")
parser = VAE.add_model_specific_args(parser)
args = parser.parse_args()

dataset = CenterOutDemonstrationDataset(
        "data/demonstration_center_out.pkl", 
        radius_cutoff=args.radius_cutoff,
        size_limit=args.size_limit)

train_set, test_set = torch.utils.data.random_split(
        dataset, [int(len(dataset) * .8), len(dataset) - int(len(dataset) * .8)])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

model = VAE(action_dim=7, **vars(args))
model.set_kl_scheduler(n_steps=args.max_epochs*len(train_loader)) 

if not args.no_wandb:
    wandb_logger = WandbLogger(
            project="latent-action", entity="ucla-ncel-robotics")
    trainer = Trainer(
            logger=wandb_logger, 
            auto_select_gpus=True,
            max_epochs=args.max_epochs)
else:
    trainer = Trainer(
            auto_select_gpus=True,
            max_epochs=args.max_epochs)

print(model)
model_summary = ModelSummary(model)

if not args.no_wandb:
    wandb_logger.log_hyperparams({
        "total_parameters": model_summary.total_parameters,
        "trainable_parameters": model_summary.trainable_parameters,
        "dataset_size": len(dataset)})

trainer.fit(model, train_loader, test_loader)
