import torch


def circular_components(deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rad = torch.deg2rad(deg)
    return torch.sin(rad), torch.cos(rad)
