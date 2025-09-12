import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(x, device=None):
    if device is None:
        device = get_device()
    if hasattr(x, "to"):
        return x.to(device)
    return x
