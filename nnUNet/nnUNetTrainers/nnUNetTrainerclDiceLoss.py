from nnunetv2.training.loss.cldice_loss import dice_cldice_loss, CE_cldice_loss, CE_clCE_loss, dice_clCE_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerDiceclDiceLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1)
        return loss

class nnUNetTrainerDiceclCELoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = dice_clCE_loss(iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1)
        return loss



class nnUNetTrainerCEclDiceLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = CE_cldice_loss({}, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1,
                              ignore_label=self.label_manager.ignore_label)
        return loss


class nnUNetTrainerCEclCEloss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = CE_clCE_loss({}, iter_=3, weight_ce=1, weight_clCE=1)
        return loss





