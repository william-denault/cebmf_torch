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
    logPhi,  # stable log CDF
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
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, omega=0.02, mu_init=1.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.pi0_head = nn.Linear(hidden_dim, 1)  # logit π0(x)

        # Raw parameter, constrained positive via softplus in forward
        mu_init = float(max(mu_init, 1e-6))
        self.mu_raw = nn.Parameter(torch.tensor(math.log(math.expm1(mu_init)), dtype=torch.float32))

        self.omega = float(omega)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        pi0 = self.sigmoid(self.pi0_head(h)).squeeze(-1)  # (N,)
        mu = self.softplus(self.mu_raw) + 1e-8            # scalar, > 0
        sigma = mu * self.omega                           # scalar
        return pi0, mu, sigma


# -------------------------
# Helpers (device/dtype-safe)
# -------------------------
def _const_like(x, value: float):
    return torch.as_tensor(value, dtype=x.dtype, device=x.device)

def _log_sqrt_2pi_like(x):
    return _const_like(x, 0.5 * math.log(2.0 * math.pi))


# -------------------------
# log N(x; m, v)  with v = variance
# -------------------------
def _log_norm_pdf(x, m, v):
    v = torch.as_tensor(v, dtype=x.dtype, device=x.device).clamp_min(1e-12)
    return -0.5 * (x - m).pow(2) / v - 0.5 * torch.log(v) - _log_sqrt_2pi_like(x)


# -------------------------
# Stable moments for TN(μ, σ^2; [0, ∞))
# -------------------------
def _tn_right0_moments(mu, sd):
    """
    Return (E[X], E[X^2]) for X ~ N(mu, sd^2) truncated to [0, ∞).
    Uses stable log-domain Mills ratio λ = φ(α) / Φ(-α), α = (0 - μ)/σ.
    """
    sd = sd.clamp_min(1e-12)
    alpha = (-mu) / sd

    log_phi = -0.5 * alpha.square() - _log_sqrt_2pi_like(mu)
    log_Z = logPhi(-alpha)  # log Φ(-α)
    # clamp to avoid inf in absurd tails
    log_lambda = (log_phi - log_Z).clamp(max=30.0)
    lam = torch.exp(log_lambda)

    EX = mu + sd * lam
    delta = lam * (lam - alpha)            # δ(α)
    var = (sd * sd) * (1.0 - delta).clamp_min(0.0)
    EX2 = var + EX.square()
    return EX, EX2


# -------------------------
# GB slab marginal: log p(x | slab) with truncation
# lg = log N(x; μ, σ^2+s^2) + log Φ(μ̃/σ̃) - log Φ(1/ω)
# -------------------------
def _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega=None):
    x = x.to(mu.dtype)
    s = s.to(mu.dtype)

    s2   = (s * s).clamp_min(1e-12)
    sig2 = (sigma * sigma).clamp_min(1e-12)
    var_sum = (s2 + sig2).clamp_min(1e-12)

    lg0 = -0.5 * (x - mu).pow(2) / var_sum - 0.5 * torch.log(var_sum) - _log_sqrt_2pi_like(x)

    inv = (1.0 / sig2) + (1.0 / s2)
    sig_tilde2 = (1.0 / inv).clamp_min(1e-12)
    sig_tilde = torch.sqrt(sig_tilde2)
    mu_tilde = sig_tilde2 * (mu / sig2 + x / s2)

    if logphi_1_over_omega is None:
        c = torch.tensor(1.0 / float(omega), dtype=mu.dtype, device=mu.device)
        logphi_1_over_omega = logPhi(c)

    lg_trunc = logPhi(mu_tilde / sig_tilde) - logphi_1_over_omega
    return lg0 + lg_trunc, mu_tilde, sig_tilde2


# -------------------------
# Mixture NLL with π0(x) and GB slab
# (returns mean NLL for minibatch training stability)
# -------------------------
def cgb_loss(pi0, x, s, mu, sigma, omega, pi0_penalty=1.0, eps=1e-12, logphi_1_over_omega=None):
    pi0 = pi0.clamp(eps, 1.0 - eps)
    s   = s.clamp_min(1e-6)

    lf = _log_norm_pdf(x, 0.0, s * s)  # spike
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)

    log_mix = torch.logaddexp(torch.log(pi0) + lf, torch.log1p(-pi0) + lg)

    if pi0_penalty != 1.0:
        # stabilise: use detached mean
        pi0_mean = pi0.mean().clamp(eps, 1.0 - eps).detach()
        log_mix = log_mix + (pi0_penalty - 1.0) * torch.log(pi0_mean)

    return -(log_mix.mean())


# -------------------------
# Responsibilities γ_i = P(slab | x_i)
# -------------------------
def gb_responsibilities(pi0, x, s, mu, sigma, omega, eps=1e-12, logphi_1_over_omega=None):
    pi0 = pi0.clamp(eps, 1.0 - eps)
    s   = s.clamp_min(1e-6)

    lf = _log_norm_pdf(x, 0.0, s * s)
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)

    log_num = torch.log1p(-pi0) + lg
    log_den = torch.logaddexp(torch.log(pi0) + lf, log_num)
    return torch.exp((log_num - log_den).clamp(min=-60.0, max=60.0)).clamp(0.0, 1.0)


# -------------------------
# Posterior moments for spike-at-0 + TN slab
# (and pointwise log p(x) for exact dataset NLL later)
# -------------------------
def gb_posterior_moments(pi0, x, s, mu, sigma, omega, logphi_1_over_omega=None):
    lg, mu_tilde, sig_tilde2 = _gb_slab_log_marginal(x, s, mu, sigma, omega, logphi_1_over_omega)
    lf = _log_norm_pdf(x, 0.0, s * s)
    gamma = gb_responsibilities(pi0, x, s, mu, sigma, omega, logphi_1_over_omega=logphi_1_over_omega)

    sd_tilde = torch.sqrt(sig_tilde2)
    EX, EX2 = _tn_right0_moments(mu_tilde, sd_tilde)

    post_mean  = gamma * EX
    post_mean2 = gamma * EX2
    post_var   = (post_mean2 - post_mean.square()).clamp_min(0.0)
    post_sd    = torch.sqrt(post_var)

    log_mix = torch.logaddexp(torch.log(pi0.clamp_min(1e-12)) + lf,
                              torch.log1p(-pi0).clamp_min(-50) + lg)

    # tiny belt-and-suspenders guard
    post_mean  = torch.nan_to_num(post_mean)
    post_mean2 = torch.nan_to_num(post_mean2)
    post_sd    = torch.nan_to_num(post_sd)

    return post_mean, post_mean2, post_sd, gamma, log_mix


# -------------------------
# Result container
# -------------------------
class CgbPosteriorResult:
    def __init__(self, post_mean, post_mean2, post_sd, pi, mu, sigma, loss, model_param, scaler):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.loss = loss  # POSITIVE scalar: dataset NLL = -∑ log p(x)
        self.model_param = model_param
        self.scaler = scaler


# -------------------------
# Main solver
# -------------------------
def sharp_cgb_posterior_means(
    X,
    betahat,
    sebetahat,
    omega=0.02,
    n_epochs=50,
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
    # Standardize X (sklearn ok here; model stays torch-only)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # move tensors to desired dtype
    dataset.betahat   = dataset.betahat.to(dtype)
    dataset.sebetahat = dataset.sebetahat.to(dtype).clamp_min(1e-6)

    # Model
    model = CgbNet(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers, omega=omega).to(dtype)
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
            se   = se.to(dtype).clamp_min(1e-6)

            pi0, mu, sigma = model(xb)
            loss = cgb_loss(
                pi0, xhat, se, mu, sigma, omega,
                pi0_penalty=pi0_penalty,
                logphi_1_over_omega=logphi_1_over_omega
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)  # accumulate NLL over samples

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs):
            with torch.no_grad():
                pi0_all, mu_all, sigma_all = model(dataset.X.to(dtype))
            print(f"[CGB] Epoch {epoch:3d}/{n_epochs} | "
                  f"Avg NLL={total_loss/len(dataset):.6f} | "
                  f"mu={mu_all.item():.4f} | sigma={sigma_all.item():.4f} | "
                  f"mean π0={pi0_all.mean().item():.4f}")

    # Posterior + exact dataset NLL on full data
    model.eval()
    with torch.no_grad():
        pi0, mu, sigma = model(dataset.X.to(dtype))
        post_mean, post_mean2, post_sd, gamma, log_mix = gb_posterior_moments(
            pi0=pi0, x=dataset.betahat, s=dataset.sebetahat, mu=mu, sigma=sigma,
            omega=omega, logphi_1_over_omega=logphi_1_over_omega
        )
        # >>> FIXED: report POSITIVE NLL over FULL dataset <<<
        nll = float((-log_mix).sum().item())

    return CgbPosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi=pi0,                     # keep π0 (spike prob)
        mu=float(mu),
        sigma=float(sigma),
        loss=nll,                   # positive scalar NLL = -∑ log p(x)
        model_param=model.state_dict(),
        scaler=scaler,
    )
