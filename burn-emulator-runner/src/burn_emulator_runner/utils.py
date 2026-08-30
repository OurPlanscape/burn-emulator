import os

_TRUTHY = ("1", "true", "yes", "on")


def env_flag(name: str) -> bool:
    """Read a boolean env var, treating only 1/true/yes/on (any case) as true."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def warm_gpu() -> None:
    import torch
    from burn_emulator.constants import RUN_DEVICE

    if RUN_DEVICE != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUN_DEVICE=cuda but torch reports no CUDA device available; "
            "check the GPU is attached and the image has the CUDA runtime"
        )
    torch.zeros(1, device="cuda")
