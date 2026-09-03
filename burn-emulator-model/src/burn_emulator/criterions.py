import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from burn_emulator.config import dynamic_import


class Reducer(nn.Module):
    def __init__(self, reduction: str) -> None:
        super().__init__()
        self.reduction = reduction
        match reduction:
            case "none":
                self.reduce = lambda inpts: inpts
            case "mean":
                self.reduce = torch.mean
            case "sum":
                self.reduce = torch.sum
            case _:
                raise ValueError(f"unknown reduction option: {reduction}")

    def forward(self, inpts: torch.Tensor) -> torch.Tensor:
        return self.reduce(inpts)


class DiceLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        multiclass: bool = False,
        epsilon: float = 1e-6,
        exclude_classes: list[int] | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.multiclass = multiclass
        self.epsilon = epsilon
        self.exclude_classes = exclude_classes
        self.label_smoothing = label_smoothing
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
        if self.exclude_classes is not None:
            keep = [c for c in range(inpts.shape[1]) if c not in self.exclude_classes]
            inpts = inpts[:, keep]
            targets = targets[:, keep]
        if self.label_smoothing:
            uniform = 1 / targets.shape[1] if self.multiclass else 0.5
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * uniform
        loss = self.forward_loss(inpts, targets)
        loss = self.reducer(loss)
        return loss


class TverskyFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | list[float] = 0.5,
        beta: float | list[float] = 0.5,
        gamma: float | list[float] = 1.0,
        reduction: str = "mean",
        multiclass: bool = False,
        epsilon: float = 1e-6,
        exclude_classes: list[int] | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))
        self.multiclass = multiclass
        self.epsilon = epsilon
        self.exclude_classes = exclude_classes
        self.label_smoothing = label_smoothing
        self.reducer = Reducer(reduction)

    def forward_loss(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tp = einsum("bcwh,bcwh->bc", inpts, targets)
        fp = einsum("bcwh,bcwh->bc", inpts, 1 - targets)
        fn = einsum("bcwh,bcwh->bc", 1 - inpts, targets)
        alpha = self.alpha.to(tp.device)
        beta = self.beta.to(tp.device)
        gamma = self.gamma.to(tp.device)
        tversky = (tp + self.epsilon) / (tp + alpha * fp + beta * fn + self.epsilon)

        return (1 - tversky) ** gamma

    def forward_activation(self, inpts: torch.Tensor) -> torch.Tensor:
        return F.softmax(inpts, dim=1) if self.multiclass else F.sigmoid(inpts)

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inpts = self.forward_activation(inpts)
        if self.exclude_classes is not None:
            keep = [c for c in range(inpts.shape[1]) if c not in self.exclude_classes]
            inpts = inpts[:, keep]
            targets = targets[:, keep]
        if self.label_smoothing:
            uniform = 1 / targets.shape[1] if self.multiclass else 0.5
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * uniform
        loss = self.forward_loss(inpts, targets)
        loss = self.reducer(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | None = None,
        reduction: str = "mean",
        multiclass: bool = True,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.multiclass = multiclass
        self.label_smoothing = label_smoothing
        self.reducer = Reducer(reduction)

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets
        if self.label_smoothing:
            uniform = 1 / targets.shape[1] if self.multiclass else 0.5
            smoothed = targets * (1 - self.label_smoothing) + self.label_smoothing * uniform

        if self.multiclass:
            ce = F.cross_entropy(inpts, smoothed, reduction="none")
            pt = torch.exp(-ce)
            loss = (1 - pt) ** self.gamma * ce
            if self.alpha is not None:
                alpha = self.alpha.to(targets.device)
                # targets are one-hot (b,c,w,h); select each pixel's
                # true-class alpha, mirroring the binary branch's alpha_t
                alpha_t = einsum("bcwh,c->bwh", targets, alpha) if alpha.ndim > 0 else alpha
                loss = alpha_t * loss
        else:
            bce = F.binary_cross_entropy_with_logits(inpts, smoothed, reduction="none")
            pt = torch.exp(-bce)
            loss = (1 - pt) ** self.gamma * bce
            if self.alpha is not None:
                alpha = self.alpha.to(targets.device)
                if alpha.ndim > 0:
                    alpha = alpha.view(1, -1, 1, 1)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss

        return self.reducer(loss)


class CompositeLoss(nn.Module):
    def __init__(self, losses: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        modules = []
        weights = []
        for spec in losses:
            weights.append(spec.get("weight", 1.0))
            loss_spec = {k: v for k, v in spec.items() if k != "weight"}
            modules.append(dynamic_import(loss_spec))
        self.losses = nn.ModuleList(modules)
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = self.weights[0] * self.losses[0](inpts, targets)
        for loss, weight in zip(self.losses[1:], self.weights[1:], strict=True):
            total = total + weight * loss(inpts, targets)
        return total


class DiceCELoss(CompositeLoss):
    def __init__(
        self,
        dice_weight: float = 0.50,
        ce_weight: float = 0.50,
        reduction: str = "mean",
        multiclass: bool = True,
        epsilon: float = 1e-6,
        exclude_classes: list[int] | None = None,
        **kwargs,
    ) -> None:
        ce_cls = "torch.nn.CrossEntropyLoss" if multiclass else "torch.nn.BCEWithLogitsLoss"
        super().__init__(
            losses=[
                {
                    "class_path": "burn_emulator.criterions.DiceLoss",
                    "init_args": dict(
                        reduction=reduction,
                        multiclass=multiclass,
                        epsilon=epsilon,
                        exclude_classes=exclude_classes,
                    ),
                    "weight": dice_weight,
                },
                {
                    "class_path": ce_cls,
                    "init_args": dict(reduction=reduction),
                    "weight": ce_weight,
                },
            ],
            **kwargs,
        )


class DiceFocalLoss(CompositeLoss):
    def __init__(
        self,
        dice_weight: float = 0.50,
        focal_weight: float = 0.50,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
        multiclass: bool = True,
        epsilon: float = 1e-6,
        exclude_classes: list[int] | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            losses=[
                {
                    "class_path": "burn_emulator.criterions.DiceLoss",
                    "init_args": dict(
                        reduction=reduction,
                        multiclass=multiclass,
                        epsilon=epsilon,
                        exclude_classes=exclude_classes,
                        label_smoothing=label_smoothing,
                    ),
                    "weight": dice_weight,
                },
                {
                    "class_path": "burn_emulator.criterions.FocalLoss",
                    "init_args": dict(
                        gamma=gamma,
                        alpha=alpha,
                        reduction=reduction,
                        multiclass=multiclass,
                        label_smoothing=label_smoothing,
                    ),
                    "weight": focal_weight,
                },
            ],
            **kwargs,
        )


class DesmoothedActivation(nn.Module):
    def __init__(
        self,
        activation: dict,
        label_smoothing: float,
        multiclass: bool = False,
        num_classes: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.activation = dynamic_import(activation)
        self.label_smoothing = label_smoothing
        self.multiclass = multiclass
        self.num_classes = num_classes

    def forward(self, inpts: torch.Tensor) -> torch.Tensor:
        p = self.activation(inpts)
        if not self.label_smoothing:
            return p
        uniform = 1 / (self.num_classes or p.shape[1]) if self.multiclass else 0.5
        p = (p - self.label_smoothing * uniform) / (1 - self.label_smoothing)
        return p.clamp(0, 1)
