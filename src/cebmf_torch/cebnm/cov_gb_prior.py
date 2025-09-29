# ============================================================
# Covariate Generalized-Binary Prior Solver (Torch-only, GPU-friendly)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.posterior import posterior_point_mass_normal


# -------------------------
# Torch StandardScaler (GPU-friendly)
# -------------------------
class TorchStandardScaler:
    def __init__(self, eps: float = 1e-12):
        self.mean_ = None
        self.scale_ = None
        self.eps = eps

    def fit(self, X: torch.Tensor):
        """
        X: (N, D) tensor on any device; population std (ddof=0).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, unbiased=False, keepdim=True)
        # avoid zeros to keep transform stable
        self.scale_ = std.clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.scale_ + self.mean_


# -------------------------
# Dataset (expects tensors already on the right device)
# -------------------------
class DensityRegressionDataset(Dataset):
    def __init__(self, X: torch.Tensor, betahat: torch.Tensor, sebetahat: torch.Tensor):
        self.X = X
        self.betahat = betahat
        self.sebetahat = sebetahat

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


# -------------------------
# MDN-like head: π₂(x) + global μ₂
# -------------------------
class CgbNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)  # logit for π₂(x)
        self.mu_2 = nn.Parameter(torch.zeros(()))  # global slab mean (scalar)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        pi_2 = self.sigmoid(self.output_layer(x)).squeeze(-1)  # (N,)
        pi_1 = 1.0 - pi_2
        return pi_1, pi_2, self.mu_2


# -------------------------
# Loss (mixture NLL, numerically stable)
# -------------------------
def cgb_loss(pi_1, pi_2, mu_2, sigma2_sq, targets, se, penalty=1.5, eps=1e-8):
    var1 = se**2
    var2 = sigma2_sq + se**2

    logp1 = -0.5 * ((targets**2) / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * (((targets - mu_2) ** 2) / var2 + torch.log(2 * torch.pi * var2))

    log_mix = torch.logaddexp(
        pi_1.clamp_min(eps).log() + logp1,
        pi_2.clamp_min(eps).log() + logp2,
    )
    if penalty > 1.0:
        # Encourage spike via average π₀; stable and simple
        log_mix = log_mix + (penalty - 1.0) * pi_1.mean().clamp_min(eps).log()
    return -log_mix.mean()


# -------------------------
# E-step responsibilities γ₂
# -------------------------
def compute_responsibilities(pi_1, pi_2, mu_2, sigma2_sq, targets, se, eps=1e-12):
    var1 = se**2
    var2 = sigma2_sq + se**2

    logp1 = -0.5 * ((targets**2) / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * (((targets - mu_2) ** 2) / var2 + torch.log(2 * torch.pi * var2))

    log_num = pi_2.clamp_min(eps).log() + logp2
    log_den = torch.logaddexp(pi_1.clamp_min(eps).log() + logp1, log_num)
    return torch.exp(log_num - log_den)


# -------------------------
# M-step for σ₂² (scalar variance)
# -------------------------
def m_step_sigma2(gamma2, mu2, targets, se):
    resid2 = (targets - mu2) ** 2
    sigma0_sq = se**2
    num = torch.sum(gamma2 * (resid2 - sigma0_sq))
    den = torch.sum(gamma2).clamp_min(1e-8)
    return num.div(den).clamp_min(1e-6)


# -------------------------
# Exact marginal log-likelihood on full data (no penalty)
# -------------------------
@torch.no_grad()
def compute_marginal_loglik_full(model, X, betahat, se, sigma2_sq, eps=1e-12):
    model.eval()
    pi1, pi2, mu2 = model(X)
    var1 = se**2
    var2 = se**2 + sigma2_sq

    logp1 = -0.5 * ((betahat**2) / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * (((betahat - mu2) ** 2) / var2 + torch.log(2 * torch.pi * var2))

    log_mix = torch.logaddexp(
        pi1.clamp_min(eps).log() + logp1,
        pi2.clamp_min(eps).log() + logp2,
    )
    return log_mix.sum()  # scalar


# -------------------------
# Result container
# -------------------------
class CgbPosteriorResult:
    def __init__(self, post_mean, post_mean2, post_sd, pi, mu_2, sigma_2, loss, model_param):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi = pi  # π₀(x): spike probability
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2
        self.loss = loss  # negative marginal log-likelihood
        self.model_param = model_param


# -------------------------
# Main solver
# -------------------------
def cgb_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=50,
    n_layers=2,
    hidden_dim=32,
    batch_size=128,
    lr=1e-3,
    penalty: float = 1.5,
    model_param=None,
    device: torch.device | None = None,
    use_amp: bool = False,
):
    """
    Torch-only pipeline; data remain on GPU when available.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Prepare tensors on device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)

    # ---- Standardize X on device
    scaler = TorchStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Dataset / DataLoader (GPU tensors → num_workers must be 0)
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- Model / Optimizer
    model = CgbNet(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    sigma2_sq = torch.tensor(1.0, dtype=torch.float32, device=device)  # slab variance

    # ---- Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, xhat, se in dataloader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pi1, pi2, mu2 = model(xb)
                gamma2 = compute_responsibilities(pi1, pi2, mu2, sigma2_sq, xhat, se)
                with torch.no_grad():
                    sigma2_sq = m_step_sigma2(gamma2, mu2, xhat, se)
                loss = cgb_loss(pi1, pi2, mu2, sigma2_sq, xhat, se, penalty=penalty)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"[CGB] Epoch {epoch + 1}/{n_epochs}, "
                f"Loss={total_loss / max(1, len(dataloader)):.4f}, "
                f"mu2={mu2.item():.3f}, sigma2={sigma2_sq.sqrt().item():.3f}"
            )

    # ---- Posterior inference on full dataset
    model.eval()
    with torch.no_grad():
        pi1_full, pi2_full, mu2_full = model(dataset.X)
        post_mean, post_var = posterior_point_mass_normal(
            betahat=dataset.betahat,
            sebetahat=dataset.sebetahat,
            pi=pi1_full,  # spike prob
            mu0=0.0,
            mu1=mu2_full.item(),
            sigma_0=sigma2_sq.sqrt().item(),
        )
        post_mean2 = post_var + post_mean**2
        post_sd = torch.sqrt(post_var.clamp_min(0.0))
        log_marginal = compute_marginal_loglik_full(
            model, X=dataset.X, betahat=dataset.betahat, se=dataset.sebetahat, sigma2_sq=sigma2_sq
        )

    return CgbPosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi=pi1_full,
        mu_2=mu2_full.item(),
        sigma_2=sigma2_sq.sqrt().item(),
        loss=-float(log_marginal.item()),
        model_param=model.state_dict(),
    )
