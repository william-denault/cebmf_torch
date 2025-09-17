 






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
# π0(x) network; global μ (>=0) ; fixed ω
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
        pi0 = self.sigmoid(self.pi0_head(h)).squeeze(-1)            # (N,)
        mu = self.softplus(self.mu_raw) + 1e-8                      # scalar, >0
        sigma = mu * self.omega                                     # scalar
        return pi0, mu, sigma


# -------------------------
# Log N(x; m, v) helper
# -------------------------
def _log_norm_pdf(x, m, v):
    v = torch.clamp(v, min=1e-12)
    return -0.5 * ((x - m) ** 2) / v - 0.5 * torch.log(2 * torch.pi * v)


# -------------------------
# GB slab marginal log p(x | slab) with truncation
# log N(x; μ, σ^2 + s^2) + log Φ( μ̃/σ̃ ) - log Φ( μ/σ )
# -------------------------
def _gb_slab_log_marginal(x, s, mu, sigma):
    var_sum = s * s + sigma * sigma
    lg0 = _log_norm_pdf(x, mu, var_sum)

    # σ̃_i^2 = (1/σ^2 + 1/s_i^2)^{-1},  μ̃_i = σ̃_i^2 * ( μ/σ^2 + x_i / s_i^2 )
    inv = 1.0 / (sigma * sigma + 1e-12) + 1.0 / (s * s)
    sig_tilde2 = 1.0 / inv
    mu_tilde = sig_tilde2 * (mu / (sigma * sigma + 1e-12) + x / (s * s))

    # truncation terms
    lg_trunc = logPhi(mu_tilde / torch.sqrt(sig_tilde2)) - logPhi(torch.tensor(float(1.0 / max(sigma / mu, 1e-12))))
    # Note: μ/σ = 1/ω ; we evaluate as scalar
    # To reduce tiny broadcast overhead, compute scalar once:
    lg_trunc = logPhi(mu_tilde / torch.sqrt(sig_tilde2)) - logPhi(torch.tensor(1.0 / float(sigma / (mu + 1e-12))))

    return lg0 + lg_trunc, mu_tilde, sig_tilde2


# -------------------------
# Mixture NLL with π0(x) and GB slab
# -------------------------
def cgb_loss(pi0, x, s, mu, sigma, pi0_penalty=1.0, eps=1e-12):
    # spike term: N(x; 0, s^2)
    lf = _log_norm_pdf(x, 0.0, s * s)

    # slab term: GB marginal
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma)

    log_mix = torch.logaddexp(torch.log(pi0.clamp_min(eps)) + lf,
                              torch.log1p(-pi0).clamp_min(-50) + lg)

    if pi0_penalty != 1.0:
        # optional regularization on mean π0 to stabilize training
        penalty_term = (pi0_penalty - 1.0) * torch.log(pi0.mean().clamp_min(eps))
        log_mix = log_mix + penalty_term

    return -(log_mix.mean())


# -------------------------
# Responsibilities γ_i = P(slab | x_i)
# -------------------------
def gb_responsibilities(pi0, x, s, mu, sigma, eps=1e-12):
    lf = _log_norm_pdf(x, 0.0, s * s)
    lg, _, _ = _gb_slab_log_marginal(x, s, mu, sigma)

    log_num = torch.log1p(-pi0).clamp_min(-50) + lg
    log_den = torch.logaddexp(torch.log(pi0.clamp_min(eps)) + lf, log_num)
    gamma = torch.exp(log_num - log_den).clamp(0.0, 1.0)
    return gamma


# -------------------------
# Posterior moments for point-mass-at-0 + truncated normal
# -------------------------
def gb_posterior_moments(pi0, x, s, mu, sigma):
    # E-step parts
    lg, mu_tilde, sig_tilde2 = _gb_slab_log_marginal(x, s, mu, sigma)
    lf = _log_norm_pdf(x, 0.0, s * s)
    gamma = gb_responsibilities(pi0, x, s, mu, sigma)

    # Slab posterior: N_+(μ̃_i, σ̃_i^2)
    a = torch.zeros_like(x)
    b = torch.full_like(x, float("inf"))

    EX = my_etruncnorm(a, b, mean=mu_tilde, sd=torch.sqrt(sig_tilde2))
    EX2 = my_e2truncnorm(a, b, mean=mu_tilde, sd=torch.sqrt(sig_tilde2))

    post_mean = gamma * EX                # spike contributes 0
    post_mean2 = gamma * EX2              # since 0^2 = 0 on spike
    post_var = torch.clamp(post_mean2 - post_mean ** 2, min=0.0)
    post_sd = torch.sqrt(post_var)

    # (Optional) marginal log-likelihood per point
    ll = torch.logaddexp(torch.log(pi0.clamp_min(1e-12)) + lf,
                         torch.log1p(-pi0).clamp_min(-50) + lg)

    return post_mean, post_mean2, post_sd, gamma, ll


# -------------------------
# Result container
# -------------------------
class CgbPosteriorResult:
    def __init__(self, post_mean, post_mean2, post_sd, pi0, mu, sigma, loss, model_state, scaler):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi0 = pi0              # π₀(x): spike prob (depends on X)
        self.mu = mu                # global μ
        self.sigma = sigma          # global σ = ω μ
        self.loss = loss
        self.model_state = model_state
        self.scaler = scaler        # for reuse on new data


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
    model_state=None,
    verbose_every=10,
):
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model
    model = CgbNet(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers, omega=omega)
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, xhat, se in dataloader:
            pi0, mu, sigma = model(xb)

            loss = cgb_loss(pi0, xhat, se, mu, sigma, pi0_penalty=pi0_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs):
            with torch.no_grad():
                pi0_all, mu_all, sigma_all = model(dataset.X)
            print(f"[CGB] Epoch {epoch:3d}/{n_epochs} | Loss={total_loss/len(dataset):.5f} | "
                  f"mu={mu_all.item():.3f} | sigma={sigma_all.item():.3f} | "
                  f"mean π0={pi0_all.mean().item():.3f}")

    # Posterior on full data
    model.eval()
    with torch.no_grad():
        pi0, mu, sigma = model(dataset.X)
        post_mean, post_mean2, post_sd, gamma, ll = gb_posterior_moments(
            pi0=pi0, x=dataset.betahat, s=dataset.sebetahat, mu=mu, sigma=sigma
        )
        total_loss = -ll.mean().item()

    return CgbPosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=1-pi0,
        mu=float(mu),
        sigma=float(sigma),
        loss=total_loss,
        model_state=model.state_dict(),
        scaler=scaler,
    )
 