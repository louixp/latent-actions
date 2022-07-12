from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from cvae.aligned_decoder import AlignedDecoder
from cvae.cae import ConditionalAE
from cvae.cvae import ConditionalVAE
from cvae.gbc import GaussianBC 
from cvae.vae import VAE
from data.center_out import CenterOutDemonstrationDataset
from data.pick_and_place import PickAndPlaceDemonstrationDataset


parser = ArgumentParser()
deocde_align_group = parser.add_mutually_exclusive_group(required=True)
deocde_align_group.add_argument("--decode", action="store_true")
deocde_align_group.add_argument("--align", action="store_true")

parser.add_argument(
        "--model_class", default="cVAE", type=str, 
        choices=["VAE", "cVAE", "cAE", "gBC"])
parser.add_argument("--batch_size", type=int, default=32)
# NOTE: Trainer.add_argparse_args(parser) kind of pollutes the 
#   hyperparameter space.
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument(
        "--dataset", required=True, choices=["pick_and_place", "center_out"])

args, _ = parser.parse_known_args()
assert(args.decode != args.align)

if args.dataset == "center_out":
    parser = CenterOutDemonstrationDataset.add_dataset_specific_args(parser)
    args, _ = parser.parse_known_args()
    dataset = CenterOutDemonstrationDataset(
        "data/demonstration_center_out.pkl",
        radius_cutoff=args.radius_cutoff,
        size_limit=args.size_limit,
        subtract_neutral_from_context=args.subtract_neutral_from_context)
elif args.dataset == "pick_and_place":
    parser = PickAndPlaceDemonstrationDataset.add_dataset_specific_args(parser)
    args, _ = parser.parse_known_args()
    dataset = PickAndPlaceDemonstrationDataset(
            "data/demonstration-7dof.pkl", 
            include_goal=args.include_goal, 
            include_joint_angles=args.include_joint_angles,
            dof=args.dof, 
            keep_success=args.keep_success, 
            size_limit=args.size_limit)

if args.model_class == "VAE":
    ModelClass = VAE
elif args.model_class == "cAE":
    ModelClass = ConditionalAE
elif args.model_class == "gBC":
    ModelClass = GaussianBC
else:
    ModelClass = ConditionalVAE

train_set, test_set = torch.utils.data.random_split(
        dataset, [int(len(dataset) * .8), len(dataset) - int(len(dataset) * .8)])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

if args.decode:
    parser = ModelClass.add_model_specific_args(parser)
    args = parser.parse_args()

    model = ModelClass(
            context_dim=dataset.get_context_dim(), 
            action_dim=dataset.get_action_dim(), 
            **vars(args))
    model.set_kl_scheduler(n_steps=args.max_epochs*len(train_loader)) 

if args.align:
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser = AlignedDecoder.add_model_specific_args(parser)
    args = parser.parse_args()
    decoder = ModelClass.load_from_checkpoint(args.checkpoint_path)
    model = AlignedDecoder(decoder=decoder, **vars(args))

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
