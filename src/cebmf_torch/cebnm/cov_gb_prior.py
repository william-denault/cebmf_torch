# ============================================================
# Covariate Generalized-Binary Prior Solver (CGB Solver, Torch-only)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.posterior import posterior_point_mass_normal
from cebmf_torch.utils.standard_scaler import standard_scale


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
# MDN Model: π₂(x) + global μ₂
# -------------------------
class CgbNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=2):
        """
        Initialize a Covariate Generalized-Binary (CGB) neural network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int, optional
            Number of hidden units in each layer (default: 32).
        n_layers : int, optional
            Number of hidden layers (default: 2).
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)  # logit for π₂(x)
        self.mu_2 = nn.Parameter(torch.tensor(0.0))  # global mean of slab

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the CGB network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        pi_1 : torch.Tensor
            Probability of spike component for each observation.
        pi_2 : torch.Tensor
            Probability of slab component for each observation.
        mu_2 : torch.Tensor
            Global mean of the slab component.
        """
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        pi_2 = self.sigmoid(self.output_layer(x)).squeeze(-1)  # (N,)
        pi_1 = 1.0 - pi_2
        return pi_1, pi_2, self.mu_2


# -------------------------
# Loss (mixture NLL, stable)
# -------------------------
def cgb_loss(pi_1, pi_2, mu_2, sigma2_sq, targets, se, penalty=1.5, eps=1e-8):
    var1 = se**2
    var2 = sigma2_sq + se**2

    logp1 = -0.5 * ((targets - 0.0) ** 2 / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * ((targets - mu_2) ** 2 / var2 + torch.log(2 * torch.pi * var2))

    log_mix = torch.logaddexp(torch.log(pi_1.clamp_min(eps)) + logp1, torch.log(pi_2.clamp_min(eps)) + logp2)
    if penalty > 1.0:
        # take mean spike prob for stability
        pi0_clamped = pi_1.mean().clamp_min(eps)
        penalty_term = (penalty - 1.0) * torch.log(pi0_clamped)
        log_mix = log_mix + penalty_term
    return -(log_mix.mean())


# -------------------------
# E-step responsibilities (γ₂)
# -------------------------
def compute_responsibilities(pi_1, pi_2, mu_2, sigma2_sq, targets, se):
    var1 = se**2
    var2 = sigma2_sq + se**2

    logp1 = -0.5 * ((targets - 0.0) ** 2 / var1 + torch.log(2 * torch.pi * var1))
    logp2 = -0.5 * ((targets - mu_2) ** 2 / var2 + torch.log(2 * torch.pi * var2))

    log_num = torch.log(pi_2.clamp_min(1e-12)) + logp2
    log_den = torch.logaddexp(torch.log(pi_1.clamp_min(1e-12)) + logp1, log_num)
    return torch.exp(log_num - log_den)


# -------------------------
# M-step for σ₂²
# -------------------------
def m_step_sigma2(gamma2, mu2, targets, se):
    resid2 = (targets - mu2) ** 2
    sigma0_sq = se**2
    num = torch.sum(gamma2 * (resid2 - sigma0_sq))
    den = torch.sum(gamma2).clamp_min(1e-8)
    return torch.clamp(num / den, min=1e-6)


# -------------------------
# Result container
# -------------------------
class CgbPosteriorResult:
    def __init__(self, post_mean, post_mean2, post_sd, pi, mu_2, sigma_2, loss, model_param):
        """
        Container for the results of the CGB posterior mean estimation.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means for each observation.
        post_mean2 : torch.Tensor
            Posterior second moments for each observation.
        post_sd : torch.Tensor
            Posterior standard deviations for each observation.
        pi : torch.Tensor
            Spike probabilities for each observation.
        mu_2 : float
            Global mean of the slab component.
        sigma_2 : float
            Global standard deviation of the slab component.
        loss : float
            Final training loss or log-likelihood.
        model_param : dict
            Trained model parameters (state_dict).
        """
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi = pi  # π₀(x): spike weight
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2
        self.loss = loss
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
):
    """
    Fit a Covariate Generalized-Binary (CGB) model to estimate the prior distribution of effects.

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
    hidden_dim : int, optional
        Number of hidden units in each layer (default=32).
    batch_size : int, optional
        Batch size for training (default=128).
    lr : float, optional
        Learning rate for the optimizer (default=1e-3).
    penalty : float, optional
        Penalty for spike probability (default=1.5).
    model_param : dict, optional
        Pre-trained model parameters to initialize the network.

    Returns
    -------
    CgbPosteriorResult
        Container with posterior means, standard deviations, and model parameters.
    """
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_scaled = standard_scale(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Init model
    model = CgbNet(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, n_layers=n_layers)
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sigma2_sq = torch.tensor(1.0, dtype=torch.float32)  # slab variance

    # Training
    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, xhat, se in dataloader:
            pi1, pi2, mu2 = model(xb)

            # E-step
            gamma2 = compute_responsibilities(pi1, pi2, mu2, sigma2_sq, xhat, se)

            # M-step
            with torch.no_grad():
                sigma2_sq = m_step_sigma2(gamma2, mu2, xhat, se)

            # Loss + update
            loss = cgb_loss(pi1, pi2, mu2, sigma2_sq, xhat, se, penalty=penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"[CGB] Epoch {epoch + 1}/{n_epochs}, Loss={total_loss / len(dataloader):.4f}, "
                f"mu2={mu2.item():.3f}, sigma2={sigma2_sq.sqrt().item():.3f}"
            )

    # Posterior inference
    model.eval()
    with torch.no_grad():
        pi1, pi2, mu2 = model(dataset.X)
        post_mean, post_var = posterior_point_mass_normal(
            betahat=dataset.betahat,
            sebetahat=dataset.sebetahat,
            pi=pi1,  # spike prob
            mu0=0.0,
            mu1=mu2.item(),
            sigma_0=sigma2_sq.sqrt().item(),
        )
        post_mean2 = post_var + post_mean**2
        post_sd = torch.sqrt(torch.clamp(post_var, min=0.0))

    return CgbPosteriorResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi=pi1,
        mu_2=mu2.item(),
        sigma_2=sigma2_sq.sqrt().item(),
        loss=total_loss,
        model_param=model.state_dict(),
    )
