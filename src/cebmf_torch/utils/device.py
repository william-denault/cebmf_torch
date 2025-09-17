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


def to_device(x: torch.Tensor, device: torch.device | None = None):
    """
    Move a tensor or module to the specified device.

    Parameters
    ----------
    x : object
        Tensor, module, or object supporting the .to() method.
    device : torch.device or None, optional
        Target device. If None, uses the default device from get_device().

    Returns
    -------
    object
        The input object moved to the specified device, or unchanged if not supported.
    """
    if device is None:
        device = get_device()
    return x.to(device)
