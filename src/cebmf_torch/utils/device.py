import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the default torch device (CUDA if available, else CPU).

    Parameters
    ----------
    prefer_cuda : bool, optional
        If True, prefer CUDA if available. Defaults to True.

    Returns
    -------
    torch.device
        The selected device ("cuda" or "cpu").
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(x, device=None):
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
    if hasattr(x, "to"):
        return x.to(device)
    return x
