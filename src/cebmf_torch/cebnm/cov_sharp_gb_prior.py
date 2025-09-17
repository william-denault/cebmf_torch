# ============================================================
# Covariate-Moderated GB Prior (π0(x) only), Trunc-Normal slab
# ============================================================

import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logPhi,
    my_etruncnorm,
    my_e2truncnorm,
)

# -------------------------
# Dataset
# -------------------------
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.betahat = torch.as_tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


# -------------------------
# π0(x) network; global μ (>=0); fixed ω
# -------------------------
class CgbNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, omega=0.2, mu_init=1.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.pi0_head = nn.Linear(hidden_dim, 1)  # logit π0(x)
        # raw parameter, constrained positive via softplus in forward
        self.mu_raw = nn.Parameter(torch.tensor(float(math.log(math.expm1(max(mu_init, 1e-6))))))

        self.omega = float(omega)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        pi0 = self.sigmoid(self.pi0_head(h)).squeeze(-1)  # (N,)
        mu = self.softplus(self.mu_raw) + 1e-8            # scalar, >0
        sigma = mu * self.omega                           # scalar
        return pi0, mu, sigma


# -------------------------
# Log N(x; m, v)
# -------------------------
def _log_norm_pdf(x, m, v):
    v = torch.clamp(v, min=1e-12)
    return -0.5 * ((x - m) ** 2) / v - 0.5 * torch.log(2 * torch.pi * v)


# -------------------------
# GB slab marginal: log p(x | slab) with truncation
# lg = log N(x; μ, σ^2+s^2) + log Φ(μ̃/σ̃) - log Φ(1/ω)
# -------------------------
def _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega=None):
    x = x.to(mu.dtype)
    s = s.to(mu.dtype)

    var_sum = torch.clamp(s * s + sigma * sigma, min=1e-12)
    lg0 = -0.5 * ((x - mu) ** 2) / var_sum - 0.5 * torch.log(2 * torch.pi * var_sum)

    inv = 1.0 / torch.clamp(sigma * sigma, min=1e-12) + 1.0 / torch.clamp(s * s, min=1e-12)
    sig_tilde2 = 1.0 / inv
    sig_tilde = torch.sqrt(torch.clamp(sig_tilde2, min=1e-12))
    mu_tilde = sig_tilde2 * (
        mu / torch.clamp(sigma * sigma, min=1e-12) + x / torch.clamp(s * s, min=1e-12)
    )

    if logphi_1_over_omega is None:
        const = torch.tensor(1.0 / float(omega), dtype=mu.dtype, device=mu.device)
        logphi_1_over_omega = logPhi(const)

    lg_trunc = logPhi(mu_tilde / sig_tilde) - logphi_1_over_omega
    return lg0 + lg_trunc, mu_tilde, sig_tilde2


# -------------------------
# Mixture NLL with π0(x) and GB slab
# -------------------------
def cgb_loss(pi0, x, s, mu, sigma, omega, pi0_penalty=1.0, eps=1e-12, logphi_1_over_omega=None):
    pi0 = pi0.clamp(eps, 1.0 - eps)
    s = torch.clamp(s, min=1e-6)

    lf = _log_norm_pdf(x, 0.0, s * s)  # spike
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)

    log_mix = torch.logaddexp(torch.log(pi0) + lf, torch.log1p(-pi0) + lg)

    if pi0_penalty != 1.0:
        # stabilise: use detached mean
        pi0_mean = pi0.mean().clamp(eps, 1.0 - eps).detach()
        log_mix = log_mix + (pi0_penalty - 1.0) * torch.log(pi0_mean)

    return -log_mix.mean()


# -------------------------
# Responsibilities γ_i = P(slab | x_i)
# -------------------------
def gb_responsibilities(pi0, x, s, mu, sigma, omega, eps=1e-12, logphi_1_over_omega=None):
    pi0 = pi0.clamp(eps, 1.0 - eps)
    s = torch.clamp(s, min=1e-6)

    lf = _log_norm_pdf(x, 0.0, s * s)
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)

    log_num = torch.log1p(-pi0) + lg
    log_den = torch.logaddexp(torch.log(pi0) + lf, log_num)
    return torch.exp(log_num - log_den).clamp(0.0, 1.0)


# -------------------------
# Posterior moments for point-mass-at-0 + truncated normal
# -------------------------
def gb_posterior_moments(pi0, x, s, mu, sigma, omega, logphi_1_over_omega=None):
    lg, mu_tilde, sig_tilde2 = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)
    lf = _log_norm_pdf(x, 0.0, s * s)
    gamma = gb_responsibilities(pi0, x, s, mu, sigma, omega, logphi_1_over_omega=logphi_1_over_omega)

    a = torch.zeros_like(x)
    b = torch.full_like(x, float("inf"))
    EX  = my_etruncnorm(a, b, mean=mu_tilde, sd=torch.sqrt(sig_tilde2))
    EX2 = my_e2truncnorm(a, b, mean=mu_tilde, sd=torch.sqrt(sig_tilde2))

    post_mean  = gamma * EX
    post_mean2 = gamma * EX2
    post_var   = torch.clamp(post_mean2 - post_mean ** 2, min=0.0)
    post_sd    = torch.sqrt(post_var)

    ll = torch.logaddexp(torch.log(pi0.clamp_min(1e-12)) + lf,
                         torch.log1p(-pi0).clamp_min(-50) + lg)
    return post_mean, post_mean2, post_sd, gamma, ll


# -------------------------
# Result container
# -------------------------
class CgbPosteriorResult:
    def __init__(self, post_mean, post_mean2, post_sd, pi0, mu, sigma, loss, model_param, scaler):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi0 = pi0
        self.mu = mu
        self.sigma = sigma
        self.loss = loss
        self.model_param = model_param
        self.scaler = scaler


# -------------------------
# Main solver
# -------------------------
def sharp_cgb_posterior_means(
    X,
    betahat,
    sebetahat,
    omega=0.2,
    n_epochs=80,
    n_layers=2,
    hidden_dim=64,
    batch_size=256,
    lr=2e-3,
    pi0_penalty: float = 1.0,
    model_param=None,
    verbose_every=10,
    dtype=torch.float64,
    grad_clip=5.0,
):
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # move tensors to desired dtype
    dataset.betahat    = dataset.betahat.to(dtype)
    dataset.sebetahat  = torch.clamp(dataset.sebetahat.to(dtype), min=1e-6)

    # Model
    model = CgbNet(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers, omega=omega)
    model = model.to(dtype)
    if model_param is not None:
        model.load_state_dict(model_param)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cache constant logΦ(1/ω)
    logphi_1_over_omega = logPhi(torch.tensor(1.0 / omega, dtype=dtype, device=dataset.betahat.device))

    # Train
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, xhat, se in dataloader:
            xb   = xb.to(dtype)
            xhat = xhat.to(dtype)
            se   = torch.clamp(se.to(dtype), min=1e-6)

            pi0, mu, sigma = model(xb)
            loss = cgb_loss(
                pi0, xhat, se, mu, sigma, omega,
                pi0_penalty=pi0_penalty,
                logphi_1_over_omega=logphi_1_over_omega
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs):
            with torch.no_grad():
                pi0_all, mu_all, sigma_all = model(dataset.X.to(dtype))
            print(f"[CGB] Epoch {epoch:3d}/{n_epochs} | Loss={total_loss/len(dataset):.6f} | "
                  f"mu={mu_all.item():.4f} | sigma={sigma_all.item():.4f} | "
                  f"mean π0={pi0_all.mean().item():.4f}")

    # Posterior on full data
    model.eval()
    with torch.no_grad():
        pi0, mu, sigma = model(dataset.X.to(dtype))
        post_mean, post_mean2, post_sd, gamma, ll = gb_posterior_moments(
            pi0=pi0, x=dataset.betahat, s=dataset.sebetahat, mu=mu, sigma=sigma,
            omega=omega, logphi_1_over_omega=logphi_1_over_omega
        )
        total_loss = -ll.mean().item()

    return CgbPosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=pi0,                     # keep π0 (spike prob)
        mu=float(mu),
        sigma=float(sigma),
        loss=total_loss,
        model_param=model.state_dict(),
        scaler=scaler,
    )
