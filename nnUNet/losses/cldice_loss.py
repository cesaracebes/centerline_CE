import torch
import torch.nn as nn
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from .soft_skeleton import soft_skel


def soft_dice(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice



class dice_cldice_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1):
        super(dice_cldice_loss, self).__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_dice=weight_dice
        self.weight_cldice=weight_cldice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:#def forward(y_true, y_pred):
        y_pred=y_pred.softmax(dim=1)

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter_)
        skel_true = soft_skel(y_true_oh, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)
        #total_loss: torch.Tensor = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        result = self.weight_dice * dice + self.weight_cldice * cl_dice
        return result


class dice_clCE_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1):
        super(dice_clCE_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.weight_clCE = weight_clCE
        self.weight_dice = weight_dice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_dice * dice + self.weight_clCE * cl_ce
        return result



#CE + algo
class CE_cldice_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1, ignore_label=None):
        super(CE_cldice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_cldice = weight_cldice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None) else 0

        target_oh = torch.zeros(net_output.shape, device=net_output.device)
        target_oh.scatter_(1, target.long(), 1) # nnUNet-training-loss-dice.py

        net_output=net_output.softmax(dim=1)
        skel_true = soft_skel(target_oh, self.iter_)
        skel_pred = soft_skel(net_output, self.iter_)
        #tprec = (torch.sum(torch.multiply(skel_pred, target[:, 0])[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tprec = (torch.sum(torch.multiply(skel_pred, target_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)

        ##Total loss computation##
        result = self.weight_ce * ce_loss + self.weight_cldice * cl_dice
        return result

class CE_clCE_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, weight_ce=1, weight_clCE=1):
        super(CE_clCE_loss, self).__init__()
        self.iter = iter_
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.weight_clCE = weight_clCE
        self.weight_ce = weight_ce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        ce_loss = self.ce(y_pred, y_true[:, 0])
        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_ce * ce_loss + self.weight_clCE * cl_ce
        return result



