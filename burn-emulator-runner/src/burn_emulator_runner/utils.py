import os

_TRUTHY = ("1", "true", "yes", "on")


def env_flag(name: str) -> bool:
    """Read a boolean env var, treating only 1/true/yes/on (any case) as true."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def warm_gpu() -> None:
    """Pay the one-time CUDA init cost at startup instead of on the first request."""
    import torch
    from burn_emulator.constants import RUN_DEVICE

    if RUN_DEVICE == "cuda" and torch.cuda.is_available():
        torch.zeros(1, device="cuda")
