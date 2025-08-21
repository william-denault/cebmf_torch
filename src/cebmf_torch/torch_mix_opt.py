
import torch
from torch import Tensor
from .torch_utils import logsumexp, safe_log, softmax

def optimize_pi_logL_torch(
    logL: Tensor,
    penalty: float = 1.0,
    batch_size: int = 65536,
    steps: int = 200,
    method: str = "adam",
    lr: float = 0.05,
    seed: int = 1234,
) -> Tensor:
    """
    Optimize mixture weights pi given per-datum log-likelihoods (n,K).
    Pure PyTorch, supports GPU and mini-batches.

    penalty: Dirichlet alpha_1 on the first component (adds (penalty-1)*log pi_0 to objective).
    method: 'adam' (recommended) or 'em'.
    """
    device = logL.device
    n, K = logL.shape
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    if method == "adam":
        logits = torch.zeros(K, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=lr)
        for t in range(steps):
            idx = torch.randint(low=0, high=n, size=(min(batch_size, n),), generator=g, device=device)
            Lb = logL.index_select(0, idx)  # (b,K)
            pi = torch.softmax(logits, dim=-1)  # (K,)
            log_pi = torch.log(pi + 1e-32)
            ll = torch.mean(logsumexp(Lb + log_pi.view(1,K), dim=1))
            reg = (penalty - 1.0) * torch.log(pi[0] + 1e-32)  # adds to objective
            loss = -(ll + reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pi = torch.softmax(logits, dim=-1)
        return pi

    elif method == "em":
        # Online EM with exponential moving average of counts
        pi = torch.full((K,), 1.0 / K, device=device)
        N = torch.zeros(K, device=device)
        gamma = min(1.0, batch_size / float(n))
        for t in range(steps):
            idx = torch.randint(low=0, high=n, size=(min(batch_size, n),), generator=None, device=device)
            Lb = logL.index_select(0, idx)  # (b,K)
            w = torch.softmax(Lb + torch.log(pi + 1e-32).view(1, K), dim=1)  # (b,K)
            Nk = w.sum(dim=0)  # (K,)
            N = (1.0 - gamma) * N + gamma * Nk
            vec_pen = torch.ones_like(N)
            vec_pen[0] = penalty
            pi = torch.clamp(N + vec_pen - 1.0, min=1e-32)
            pi = pi / pi.sum()
        return pi
    else:
        raise ValueError("method must be 'adam' or 'em'")
