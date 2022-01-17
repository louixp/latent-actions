from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from cvae.vae import VAE
from cvae.cvae import ConditionalVAE
from cvae.dataset import DemonstrationDataset

parser = ArgumentParser()
parser.add_argument(
        "--model_class", default="cVAE", type=str, 
        choices=["VAE", "cVAE", "gBC"])
script_args, _ = parser.parse_known_args()
if script_args.model_class == "VAE":
    ModelClass = VAE
elif script_args.model_class == "gBC":
    ModelClass = GaussianBC
else:
    ModelClass = ConditionalVAE

parser.add_argument("--batch_size", type=int, default=32)
# NOTE: Trainer.add_argparse_args(parser) kind of pollutes the 
#   hyperparameter space.
parser.add_argument("--max_epochs", type=int, default=400)
parser = DemonstrationDataset.add_dataset_specific_args(parser)
parser = ModelClass.add_model_specific_args(parser)
args = parser.parse_args()

dataset = DemonstrationDataset.from_baselines_rl_zoo(
        "../rl-baselines3-zoo/demonstration.pkl", args.include_goal)
train_set, test_set = torch.utils.data.random_split(
        dataset, [int(len(dataset) * .8), int(len(dataset) * .2)])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

wandb_logger = WandbLogger(project="latent-action", entity="ucla-ncel-robotics")
trainer = Trainer(
        logger=wandb_logger, 
        auto_select_gpus=True,
        max_epochs=args.max_epochs)
    
model = ModelClass(
        context_dim=dataset.get_context_dim(), 
        action_dim=dataset.get_action_dim(), 
        **vars(args))
model.set_kl_scheduler(n_steps=trainer.max_epochs*len(train_loader)) 

print(model)
model_summary = ModelSummary(model)
wandb_logger.log_hyperparams({
    "total_parameters": model_summary.total_parameters,
    "trainable_parameters": model_summary.trainable_parameters,
    "dataset_size": len(dataset)})

trainer.fit(model, train_loader, test_loader)
