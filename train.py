from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from latent_actions import cvae, data


parser = ArgumentParser()
parser.add_argument(
        "--model_class", default="cVAE", type=str, 
        choices=cvae.DECODER_CLASS.keys())
parser.add_argument("--batch_size", type=int, default=32)
# NOTE: Trainer.add_argparse_args(parser) kind of pollutes the 
#   hyperparameter space.
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument(
        "--dataset", required=True, choices=data.DATASET_CLASS.keys())
args, _ = parser.parse_known_args()

DataClass = data.DATASET_CLASS[args.dataset]
ModelClass = cvae.DECODER_CLASS[args.model_class]

parser = DataClass.add_dataset_specific_args(parser)
parser = ModelClass.add_model_specific_args(parser)
args = parser.parse_args()

dataset = DataClass(**vars(args))
train_set, test_set = torch.utils.data.random_split(
        dataset, [int(len(dataset) * .8), len(dataset) - int(len(dataset) * .8)])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

model = ModelClass(
        context_dim=dataset.get_context_dim(), 
        action_dim=dataset.get_action_dim(), 
        **vars(args))
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
