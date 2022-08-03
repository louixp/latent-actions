from latent_actions.data.center_out import CenterOutDemonstrationDataset
from latent_actions.data.real_center_out import RealCenterOutDemonstrationDataset
from latent_actions.data.pick_and_place import PickAndPlaceDemonstrationDataset

DATASET_CLASS = {
        "center_out": CenterOutDemonstrationDataset,
        "real_center_out": RealCenterOutDemonstrationDataset,
        "pick_and_place": PickAndPlaceDemonstrationDataset}
