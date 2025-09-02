
import torch
import math

def autoselect_scales_mix_norm(betahat: torch.Tensor, sebetahat: torch.Tensor, max_class=None, mult: float = 2.0):
    device = betahat.device
    sigmaamin = torch.min(sebetahat) / 10.0
    if torch.all(betahat**2 < sebetahat**2):
        sigmaamax = 8.0 * sigmaamin
    else:
        sigmaamax = 2.0 * torch.sqrt(torch.max(betahat**2 - sebetahat**2))

    if mult == 0:
        out = torch.tensor([0.0, sigmaamax / 2.0], device=device)
    else:
        npoint = int(math.ceil(float(torch.log2(sigmaamax / sigmaamin)) / math.log2(mult)))
        seq = torch.arange(-npoint, 1, device=device, dtype=torch.int64)
        out = torch.cat([torch.tensor([0.0], device=device), (1.0/mult) ** (-seq.float()) * sigmaamax])
        if max_class is not None:
            if out.numel() != max_class:
                out = torch.linspace(torch.min(out), torch.max(out), steps=max_class, device=device)
    return out

def autoselect_scales_mix_exp(betahat: torch.Tensor, sebetahat: torch.Tensor, max_class=None, mult: float = 1.5, tt: float = 1.5):
    device = betahat.device
    sigmaamin = torch.maximum(torch.min(sebetahat) / 10.0, torch.tensor(1e-3, device=device))
    if torch.all(betahat**2 < sebetahat**2):
        sigmaamax = 8.0 * sigmaamin
    else:
        sigmaamax = tt * torch.sqrt(torch.max(betahat**2))

    if mult == 0:
        out = torch.tensor([0.0, sigmaamax / 2.0], device=device)
    else:
        npoint = int(math.ceil(float(torch.log2(sigmaamax / sigmaamin)) / math.log2(mult)))
        seq = torch.arange(-npoint, 1, device=device, dtype=torch.int64)
        out = torch.cat([torch.tensor([0.0], device=device), (1.0/mult) ** (-seq.float()) * sigmaamax])
        if max_class is not None:
            if out.numel() != max_class:
                out = torch.linspace(torch.min(out), torch.max(out), steps=max_class, device=device)
                if out.numel() >= 3 and out[2] < 1e-2:
                    out[2:] = out[2:] + 1e-2
    return out
