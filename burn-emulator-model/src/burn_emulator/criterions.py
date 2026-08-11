import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum


class Reducer(nn.Module):
    def __init__(self, reduction: str) -> None:
        super().__init__()
        self.reduction = reduction
        match reduction:
            case "none":
                self.reduce = lambda inputs: inputs
            case "mean":
                self.reduce = torch.mean
            case "sum":
                self.reduce = torch.sum
            case _:
                raise ValueError(f"unknown reduction option: {reduction}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reduce(inputs)


class DiceLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        multiclass: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.multiclass = multiclass
        self.epsilon = epsilon
        self.reducer = Reducer(reduction)

    def forward_loss(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = einsum("bcwh,bcwh->bc", inpts, targets)
        sum_probs = einsum("bcwh->bc", inpts) + einsum("bcwh->bc", targets)
        loss = (2.0 * intersection + self.epsilon) / (sum_probs + self.epsilon)

        return 1 - loss

    def forward_activation(self, inpts: torch.Tensor) -> torch.Tensor:
        return F.softmax(inpts, dim=1) if self.multiclass else F.sigmoid(inpts)

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inpts = self.forward_activation(inpts)
        loss = self.forward_loss(inpts, targets)
        loss = self.reducer(loss)
        return loss


class DiceCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = .5,
        ce_weight: float = .5,
        reduction: str = "mean",
        multiclass: bool = True,
        epsilon: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(
            reduction=reduction,
            multiclass=multiclass,
            epsilon=epsilon,
        )
        self.ce_loss = (    
            nn.CrossEntropyLoss(reduction=reduction)
            if multiclass
            else nn.BCEWithLogitsLoss(reduction=reduction)
        )

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(inpts, targets)
        ce = self.ce_loss(inpts, targets)
        return self.dice_weight * dice + self.ce_weight * ce
