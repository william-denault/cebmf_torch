# ============================================================
# Two-sided Covariate Sharp Generalized-Binary Prior Solver
# (CGB-Sharp-2 Solver, Torch-only)
#
# Prior model (per observation x with covariate x):
#
#     theta ~ pi_0(x) * delta_0
#           + pi_1(x) * N(mu_+, omega * sigma_1^2)
#           + pi_2(x) * N(mu_-, omega * sigma_2^2)
#
# with mu_+ >= 0 (positive slab) and mu_- <= 0 (negative slab) -- both
# may be exactly zero. The point-mass-enforcing penalty on pi_0 from
# the original CGB-Sharp solver is preserved.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.posterior import posterior_mean_norm
from cebmf_torch.utils.standard_scaler import standard_scale


# -------------------------
# Dataset
# -------------------------
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = X
        self.betahat = betahat
        self.sebetahat = sebetahat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


# -------------------------
# MDN model: pi_0(x), pi_1(x), pi_2(x) via softmax + global mu_+, mu_-
# -------------------------
class Cgb2Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=2):
        """
        Initialize the two-sided Covariate Sharp Generalized-Binary network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int, optional
            Number of hidden units per layer (default: 32).
        n_layers : int, optional
            Number of hidden layers (default: 2).
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        # 3 logits -> softmax over (spike, pos-slab, neg-slab)
        self.output_layer = nn.Linear(hidden_dim, 3)
        # Global slab means; transformed via softplus to enforce sign while
        # still allowing 0 (softplus is C^infty and bounded below by 0).
        self.raw_mu_pos = nn.Parameter(torch.tensor(0.0))
        self.raw_mu_neg = nn.Parameter(torch.tensor(0.0))

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        pi_0 : torch.Tensor
            Spike probabilities for each observation, shape (N,).
        pi_1 : torch.Tensor
            Positive-slab probabilities, shape (N,).
        pi_2 : torch.Tensor
            Negative-slab probabilities, shape (N,).
        mu_pos : torch.Tensor
            Global mean of the positive slab, >= 0.
        mu_neg : torch.Tensor
            Global mean of the negative slab, <= 0.
        """
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        logits = self.output_layer(x)  # (N, 3)
        pi = torch.softmax(logits, dim=-1)  # (N, 3)
        pi_0 = pi[..., 0]
        pi_1 = pi[..., 1]
        pi_2 = pi[..., 2]
        mu_pos = self.softplus(self.raw_mu_pos)
        mu_neg = -self.softplus(self.raw_mu_neg)
        return pi_0, pi_1, pi_2, mu_pos, mu_neg


# -------------------------
# Loss (mixture NLL, stable)
# -------------------------
def cgb2_loss(
    pi_0,
    pi_1,
    pi_2,
    mu_pos,
    mu_neg,
    sigma1_sq,
    sigma2_sq,
    targets,
    se,
    penalty=1.5,
    eps=1e-8,
):
    """
    Negative log marginal likelihood of the 3-component mixture, plus an
    optional point-mass-enforcing penalty on pi_0(x).
    """
    var0 = se**2
    var1 = sigma1_sq + se**2
    var2 = sigma2_sq + se**2

    logp0 = -0.5 * ((targets - 0.0) ** 2 / var0 + torch.log(2 * torch.pi * var0))
    logp1 = -0.5 * ((targets - mu_pos) ** 2 / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * ((targets - mu_neg) ** 2 / var2 + torch.log(2 * torch.pi * var2))

    # stable log mixture across (spike, pos-slab, neg-slab)
    log_terms = torch.stack(
        [
            torch.log(pi_0.clamp_min(eps)) + logp0,
            torch.log(pi_1.clamp_min(eps)) + logp1,
            torch.log(pi_2.clamp_min(eps)) + logp2,
        ],
        dim=-1,
    )
    log_mix = torch.logsumexp(log_terms, dim=-1)

    if penalty > 1.0:
        # Penalize per-observation spike probability (Dirichlet-like prior on
        # component 0). Same shape/behavior as the single-slab CGB-Sharp.
        log_pi0 = torch.log(pi_0.clamp_min(eps))
        log_mix = log_mix + (penalty - 1.0) * log_pi0
    return -(log_mix.mean())


# -------------------------
# E-step responsibilities for the two slab components (gamma_1, gamma_2)
# -------------------------
def compute_responsibilities(
    pi_0,
    pi_1,
    pi_2,
    mu_pos,
    mu_neg,
    sigma1_sq,
    sigma2_sq,
    targets,
    se,
    eps=1e-12,
):
    var0 = se**2
    var1 = sigma1_sq + se**2
    var2 = sigma2_sq + se**2

    logp0 = -0.5 * ((targets - 0.0) ** 2 / var0 + torch.log(2 * torch.pi * var0))
    logp1 = -0.5 * ((targets - mu_pos) ** 2 / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * ((targets - mu_neg) ** 2 / var2 + torch.log(2 * torch.pi * var2))

    log_terms = torch.stack(
        [
            torch.log(pi_0.clamp_min(eps)) + logp0,
            torch.log(pi_1.clamp_min(eps)) + logp1,
            torch.log(pi_2.clamp_min(eps)) + logp2,
        ],
        dim=-1,
    )
    log_den = torch.logsumexp(log_terms, dim=-1, keepdim=True)
    resp = torch.exp(log_terms - log_den)
    return resp[..., 0], resp[..., 1], resp[..., 2]  # gamma_0, gamma_1, gamma_2


# -------------------------
# M-step for a single slab variance, given its responsibilities + slab mean
# -------------------------
def m_step_sigma(gamma, mu, targets, se):
    resid2 = (targets - mu) ** 2
    sigma0_sq = se**2
    num = torch.sum(gamma * (resid2 - sigma0_sq))
    den = torch.sum(gamma).clamp_min(1e-8)
    return torch.clamp(num / den, min=1e-6)


# -------------------------
# Result container
# -------------------------
class Cgb2PosteriorResult:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        pi,
        pi_pos,
        pi_neg,
        mu_pos,
        mu_neg,
        sigma_1,
        sigma_2,
        loss,
        model_param,
    ):
        """
        Container for the results of the two-sided CGB-Sharp posterior estimation.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means, shape (N,).
        post_mean2 : torch.Tensor
            Posterior second moments, shape (N,).
        post_sd : torch.Tensor
            Posterior standard deviations, shape (N,).
        pi : torch.Tensor
            Spike probabilities pi_0(x) for each observation.
        pi_pos : torch.Tensor
            Positive-slab probabilities pi_1(x).
        pi_neg : torch.Tensor
            Negative-slab probabilities pi_2(x).
        mu_pos : float
            Global mean of the positive slab (>= 0).
        mu_neg : float
            Global mean of the negative slab (<= 0).
        sigma_1 : float
            Global SD of the positive slab.
        sigma_2 : float
            Global SD of the negative slab.
        loss : float
            Final (negative) marginal log-likelihood.
        model_param : dict
            Trained model state_dict.
        """
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi = pi  # spike weight, mirrors CGB / CGB-Sharp convention
        self.pi_pos = pi_pos
        self.pi_neg = pi_neg
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.loss = loss
        self.model_param = model_param


@torch.no_grad()
def compute_marginal_loglik_full(
    model,
    X,
    betahat,
    se,
    sigma1_sq,
    sigma2_sq,
    eps=1e-12,
):
    """
    Exact marginal log-likelihood for current parameters (no penalty), computed
    over the full dataset rather than per-batch.
    """
    model.eval()
    pi0, pi1, pi2, mu_pos, mu_neg = model(X)

    var0 = se**2
    var1 = se**2 + sigma1_sq
    var2 = se**2 + sigma2_sq

    logp0 = -0.5 * ((betahat - 0.0) ** 2 / var0 + torch.log(2 * torch.pi * var0))
    logp1 = -0.5 * ((betahat - mu_pos) ** 2 / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * ((betahat - mu_neg) ** 2 / var2 + torch.log(2 * torch.pi * var2))

    log_terms = torch.stack(
        [
            pi0.clamp_min(eps).log() + logp0,
            pi1.clamp_min(eps).log() + logp1,
            pi2.clamp_min(eps).log() + logp2,
        ],
        dim=-1,
    )
    log_mix = torch.logsumexp(log_terms, dim=-1)
    return log_mix.sum()


# -------------------------
# Main solver
# -------------------------
def sharp_2cgb_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=50,
    n_layers=2,
    omega=0.001,
    hidden_dim=32,
    batch_size=128,
    lr=1e-3,
    penalty: float = 1.5,
    model_param=None,
    eps=1e-8,
    device: torch.device | None = None,
):
    """
    Fit a two-sided Covariate Sharp Generalized-Binary (CGB-Sharp-2) model
    estimating the prior

        pi_0(x) * delta_0 + pi_1(x) * N(mu_+, omega * sigma_1^2)
                          + pi_2(x) * N(mu_-, omega * sigma_2^2)

    with mu_+ >= 0 and mu_- <= 0.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Covariates for each observation, shape (n_samples, n_features).
    betahat : torch.Tensor or np.ndarray
        Observed effect estimates, shape (n_samples,).
    sebetahat : torch.Tensor or np.ndarray
        Standard errors of the effect estimates, shape (n_samples,).
    n_epochs : int, optional
        Number of training epochs (default=50).
    n_layers : int, optional
        Number of hidden layers in the neural network (default=2).
    omega : float, optional
        Variance-shrinkage factor applied to the M-step slab variances
        (default=0.005); mirrors the single-slab CGB-Sharp behavior.
    hidden_dim : int, optional
        Number of hidden units in each layer (default=32).
    batch_size : int, optional
        Batch size for training (default=128).
    lr : float, optional
        Learning rate for the optimizer (default=1e-3).
    penalty : float, optional
        Penalty for spike probability pi_0 (default=1.5). Values > 1 enforce
        a point-mass-favoring Dirichlet-like prior on component 0.
    model_param : dict, optional
        Pre-trained model parameters to initialize the network.
    device : torch.device, optional
        Target device. Inherited from ``betahat`` when ``None``.

    Returns
    -------
    Cgb2PosteriorResult
        Container with posterior means, standard deviations, per-observation
        spike / slab weights, and the trained network state_dict.
    """

    # Inherit device from input tensor when available; avoids silent device
    # hops if the caller is on CPU/MPS but CUDA is also visible.
    if device is None:
        device = (
            betahat.device
            if isinstance(betahat, torch.Tensor)
            else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )

    # ---- to tensor on device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # ---- scale on device
    X_scaled = standard_scale(X)  # stays on device

    # ---- dataset / loader (GPU tensors, keep num_workers=0)
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- model / optimizer on device
    model = Cgb2Net(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sigma1_sq = torch.tensor(1.0, dtype=torch.float32, device=device)
    sigma2_sq = torch.tensor(1.0, dtype=torch.float32, device=device)

    # ---- training (per-batch E + sharp M-step, matching cov_sharp_gb_prior)
    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, xhat, se in dataloader:  # already device tensors
            pi0, pi1, pi2, mu_pos, mu_neg = model(xb)
            with torch.no_grad():
                _, gamma1, gamma2 = compute_responsibilities(
                    pi0, pi1, pi2, mu_pos, mu_neg, sigma1_sq, sigma2_sq, xhat, se
                )
                sigma1_sq = m_step_sigma(gamma1, mu_pos, xhat, se) * omega
                sigma2_sq = m_step_sigma(gamma2, mu_neg, xhat, se) * omega
            loss = cgb2_loss(
                pi0, pi1, pi2, mu_pos, mu_neg, sigma1_sq, sigma2_sq, xhat, se, penalty=penalty
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"[CGB-2] Epoch {epoch + 1}/{n_epochs}, "
                f"Loss={total_loss / len(dataloader):.4f}, "
                f"mu_pos={mu_pos.item():.3f}, mu_neg={mu_neg.item():.3f}, "
                f"sigma1={sigma1_sq.sqrt().item():.3f}, sigma2={sigma2_sq.sqrt().item():.3f}"
            )

    # ---- posterior inference (use the shared mixture-of-normals helper, with
    # the spike encoded as scale=0 at location 0).
    model.eval()
    with torch.no_grad():
        pi0, pi1, pi2, mu_pos, mu_neg = model(dataset.X)
        J = dataset.betahat.shape[0]
        K = 3
        dt = dataset.betahat.dtype
        dev = dataset.betahat.device

        sigma1 = sigma1_sq.sqrt()
        sigma2 = sigma2_sq.sqrt()

        # (J, K) per-observation log-pi
        log_pi = torch.log(
            torch.stack(
                [pi0.clamp_min(eps), pi1.clamp_min(eps), pi2.clamp_min(eps)],
                dim=-1,
            )
        )

        # Per-observation scales (spike = 0, slabs = sigma_1, sigma_2)
        zeros = torch.zeros(J, dtype=dt, device=dev)
        scale = torch.stack(
            [zeros, sigma1.expand(J), sigma2.expand(J)], dim=-1
        )  # (J, K)
        location = torch.stack(
            [zeros, mu_pos.expand(J), mu_neg.expand(J)], dim=-1
        )  # (J, K)

        # Component-wise marginal log-likelihoods log p(x | k)
        s2 = dataset.sebetahat**2
        var0 = s2
        var1 = s2 + sigma1_sq
        var2 = s2 + sigma2_sq

        logp0 = -0.5 * ((dataset.betahat - 0.0) ** 2 / var0 + torch.log(2 * torch.pi * var0))
        logp1 = -0.5 * ((dataset.betahat - mu_pos) ** 2 / var1 + torch.log(2 * torch.pi * var1))
        logp2 = -0.5 * ((dataset.betahat - mu_neg) ** 2 / var2 + torch.log(2 * torch.pi * var2))
        data_loglik = torch.stack([logp0, logp1, logp2], dim=-1)  # (J, K)

        pm = posterior_mean_norm(
            betahat=dataset.betahat,
            sebetahat=dataset.sebetahat,
            log_pi=log_pi,
            data_loglik=data_loglik,
            scale=scale,
            location=location,
        )

        post_mean = pm.post_mean
        post_mean2 = pm.post_mean2
        post_sd = pm.post_sd

        log_marginal = compute_marginal_loglik_full(
            model,
            X=dataset.X,
            betahat=dataset.betahat,
            se=dataset.sebetahat,
            sigma1_sq=sigma1_sq,
            sigma2_sq=sigma2_sq,
        )

    return Cgb2PosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi=pi0,
        pi_pos=pi1,
        pi_neg=pi2,
        mu_pos=mu_pos.item(),
        mu_neg=mu_neg.item(),
        sigma_1=sigma1_sq.sqrt().item(),
        sigma_2=sigma2_sq.sqrt().item(),
        loss=-float(log_marginal.item()),
        model_param=model.state_dict(),
    )
