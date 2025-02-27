import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from typing import Union, List
import numpy as np

class MultiTaskLossAsym(nn.Module):
    def __init__(self,
                seg_loss,
                seg_weight: float = 1.0,
                cls_weight: float = 1.0,
                deep_supervision: bool = True,
                ds_weights: List[float] = None,
                consistency: bool = False
                ) -> None:
        super().__init__()
        self.seg_loss = seg_loss
        self.deep_supervision = deep_supervision
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.consistency = consistency

        if deep_supervision:
            self.seg_loss = DeepSupervisionWrapper(self.seg_loss, 
                                                   weight_factors=ds_weights)

        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                cls_output: torch.Tensor, 
                cls_target: torch.Tensor, 
                seg_output: Union[torch.Tensor, List[torch.Tensor]],
                seg_target: Union[torch.Tensor, List[torch.Tensor]]
                ):
        
        loss_dict = {
            "seg_loss": None,
            "cls_loss": None,
            "consistency_loss": None
           }
        
        if cls_target.dim() == 2:
            cls_target = cls_target.squeeze(1)
            
        loss = self.cls_weight * self.cls_loss(cls_output, cls_target)
        loss_dict["cls_loss"] = loss.item()

        cls_output = cls_output[:, 1]

        if self.consistency:
            if self.deep_supervision:
                seg_out_temp = seg_output[0][:, 1]
            else:
                seg_out_temp = seg_output[:, 1]

            seg_out_temp = torch.max(seg_out_temp.view(seg_out_temp.size(0), -1), dim=1)[0]
            consistency_loss = F.l1_loss(seg_out_temp, cls_output.view(-1)) * 0.1

            loss_dict["consistency_loss"] = consistency_loss.item()
            loss += consistency_loss

        # get only positive seg_target
        positive_idxs = torch.where(cls_target>0)[0]

        if positive_idxs.numel() > 0:
            if self.deep_supervision:
                seg_output = [i[positive_idxs] for i in seg_output]
                seg_target = [i[positive_idxs] for i in seg_target]
            else:
                seg_target = seg_target[positive_idxs]
                seg_output = seg_output[positive_idxs]
            
            seg_loss = self.seg_loss(seg_output, seg_target)
            loss_dict["seg_loss"] = seg_loss.item()
            loss += self.seg_weight * seg_loss

        return loss, loss_dict
    
class MultiTaskLoss(nn.Module):
    def __init__(self,
                seg_loss,
                seg_weight: float = 0.5,
                cls_weight: float = 1.0,
                deep_supervision: bool = True,
                ds_weights: List[float] = None,
                consistency: bool = False
                ) -> None:
        super().__init__()
        self.seg_loss = seg_loss
        self.deep_supervision = deep_supervision
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.consistency = consistency

        if deep_supervision:
            self.seg_loss = DeepSupervisionWrapper(self.seg_loss, 
                                                   weight_factors=ds_weights)

        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                cls_output: torch.Tensor, 
                cls_target: torch.Tensor, 
                seg_output: Union[torch.Tensor, List[torch.Tensor]],
                seg_target: Union[torch.Tensor, List[torch.Tensor]]
                ):
        
        loss_dict = {
            "seg_loss": None,
            "cls_loss": None,
            "consistency_loss": None
           }
        
        if cls_target.dim() == 2:
            cls_target = cls_target.squeeze(1)
            
        loss = self.cls_weight * self.cls_loss(cls_output, cls_target)
        loss_dict["cls_loss"] = loss.item()

        cls_output = cls_output[:, 1]

        if self.consistency:
            if self.deep_supervision:
                seg_out_temp = seg_output[0][:, 1]
            else:
                seg_out_temp = seg_output[:, 1]

            seg_out_temp = torch.max(seg_out_temp.view(seg_out_temp.size(0), -1), dim=1)[0]
            consistency_loss = F.l1_loss(seg_out_temp, cls_output.view(-1)) * 0.1

            loss_dict["consistency_loss"] = consistency_loss.item()
            loss += consistency_loss

        # get only positive seg_target
        seg_loss = self.seg_loss(seg_output, seg_target)
        loss_dict["seg_loss"] = seg_loss.item()
        loss += self.seg_weight * seg_loss

        return loss, loss_dict
    