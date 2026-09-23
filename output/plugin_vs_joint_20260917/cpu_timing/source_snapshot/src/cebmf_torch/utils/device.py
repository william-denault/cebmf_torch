"""Validation that does not synchronize CUDA tensors with Python."""

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Choose CUDA, then MPS, then CPU (or CPU when prefer_gpu=False)."""
    if not prefer_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(x: torch.Tensor, device: torch.device | None = None):
    """Move a tensor or module to the chosen device."""
    return x.to(get_device() if device is None else device)


def require_finite(value, message, error_type=FloatingPointError):
    valid = torch.isfinite(value).all()
    if value.is_cuda:
        # Device-side assertion: no .item()/bool()/CPU transfer in fitting.
        # Like other CUDA assertions, a failure is reported asynchronously.
        torch._assert_async(valid, message)
    elif not valid:
        raise error_type(message)
