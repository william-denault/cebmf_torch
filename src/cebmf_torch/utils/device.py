import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device.

    Priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (fallback)

    Args:
        prefer_gpu: Whether to prefer GPU over CPU

    Returns:
        torch.device: The selected device
    """
    if not prefer_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(x: torch.Tensor, device=None):
    """Move tensor or object to specified device."""
    if device is None:
        device = get_device()
    return x.to(device)
