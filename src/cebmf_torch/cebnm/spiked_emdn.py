import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.posterior import posterior_mean_norm


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
# Mixture Density Network (spike + slabs)
#   pi: (N, K) for [spike, slabs...]
#   mu/log_sigma: (N, K-1) for slabs only
# -------------------------
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians, n_layers=4):
        super().__init__()
        assert n_gaussians >= 2, "Need at least 1 spike + 1 slab."
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.pi = nn.Linear(hidden_dim, n_gaussians)  # includes spike (k=0)
        self.mu = nn.Linear(hidden_dim, n_gaussians - 1)  # slabs only
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians - 1)  # slabs only
        self.point_mass = 0.0

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        pi = torch.softmax(self.pi(x), dim=1)  # (N, K)
        mu = self.mu(x)  # (N, K-1)
        # keep slabs' std positive and stable
        log_sigma = torch.log(torch.nn.functional.softplus(self.log_sigma(x)) + 1e-6)  # (N, K-1)
        return pi, mu, log_sigma


# -------------------------
# Loss: correct spike+slabs mixture + steerable spike penalty
# -------------------------
def mdn_spike_loss_with_varying_noise(
    pi,
    mu,
    log_sigma,
    betahat,
    sebetahat,
    *,
    penalty: float = 1.0,
    beta_prior: tuple | None = None,
    eps: float = 1e-8,
):
    # Spike likelihood: mean=0, total var = se^2
    var_spike = sebetahat**2  # (N,)
    logp_spike = -0.5 * ((betahat**2) / var_spike + torch.log(2 * torch.pi * var_spike))  # (N,)

    # Slab likelihoods: mean=mu_j, total sd = sqrt(prior_sd^2 + se^2)
    sigma_slab = torch.exp(log_sigma)  # (N, K-1) prior sd
    total_sigma_slab = torch.sqrt(sigma_slab**2 + sebetahat.unsqueeze(1) ** 2)  # (N, K-1)
    dist_slab = torch.distributions.Normal(mu, total_sigma_slab)
    logp_slabs = dist_slab.log_prob(betahat.unsqueeze(1))  # (N, K-1)

    # Mixture log-likelihood = logsumexp over [spike, slabs...]
    log_terms_spike = torch.log(pi[:, :1].clamp_min(eps)) + logp_spike.unsqueeze(1)  # (N, 1)
    log_terms_slabs = torch.log(pi[:, 1:].clamp_min(eps)) + logp_slabs  # (N, K-1)
    all_log_terms = torch.cat([log_terms_spike, log_terms_slabs], dim=1)  # (N, K)
    nll = -torch.logsumexp(all_log_terms, dim=1).mean()

    # (A) simple steer: penalty>1 encourages spike
    reg_simple = 0.0
    if penalty != 1.0:
        lam = float(penalty) - 1.0  # >0 encourages spike
        reg_simple = -(lam) * torch.log(pi[:, 0].clamp_min(eps)).mean()

    # (B) optional Beta(alpha0, beta0) prior on pi_spike
    reg_beta = 0.0
    if beta_prior is not None:
        a0, b0 = map(float, beta_prior)
        # log(1 - pi0) needs safe clamp
        one_minus_pi0 = (1.0 - pi[:, 0]).clamp_min(eps)
        reg_beta = -((a0 - 1.0) * torch.log(pi[:, 0].clamp_min(eps)) + (b0 - 1.0) * torch.log(one_minus_pi0)).mean()

    return nll + reg_simple + reg_beta


# -------------------------
# Result container
# -------------------------
class EmdnPosteriorMeanNorm:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        location,
        pi_np,
        scale,
        loss=0,
        model_param=None,
    ):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.location = location
        self.pi_np = pi_np
        self.scale = scale
        self.loss = loss
        self.model_param = model_param


# -------------------------
# Main solver
# -------------------------
def spiked_emdn_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=50,
    n_layers=4,
    n_gaussians=5,
    hidden_dim=64,
    batch_size=512,
    lr=1e-3,
    model_param=None,
    *,
    penalty: float = 1.0,  # >1 encourages spike; =1 neutral
    beta_prior: tuple | None = None,  # e.g. (17., 5.) => target pi_spike ~ 0.77
    print_every=10,
):
    """
    Fit a Mixture Density Network to estimate the prior distribution of effects.
    In the EBNM problem, we observe estimates `betahat` with standard errors `sebetahat` and want to estimate
    the prior distribution of the true effects.

    betahat ~ N(theta, sebetahat^2)

    theta ~ G, where G is modeled as a mixture of Gaussians with parameters predicted by a neural network,
    but with an additional point mass at 0 (spike + slabs).
    
    Args:
        :X (torch.Tensor): Covariates for each observation, shape (n_samples, n_features).
        :betahat (torch.Tensor): Observed effect estimates, shape (n_samples,).
        :sebetahat (torch.Tensor): Standard errors of the effect estimates, shape (n_samples,).
        :n_epochs (int): Number of training epochs.
        :n_layers (int): Number of hidden layers in the neural network.
        :n_gaussians (int): Number of Gaussian components in the mixture.
        :hidden_dim (int): Number of hidden units in each layer.
        :batch_size (int): Batch size for training.
        :lr (float): Learning rate for the optimizer.
        :model_param (dict, optional): Pre-trained model parameters to initialize the network.
        :penalty (float): >1 encourages spike; =1 neutral.
        :beta_prior (tuple, optional): (alpha0, beta0) for Beta prior on pi_spike.
        :print_every (int): Print training loss every this many epochs.

    Returns:
        :EmdnPosteriorMeanNorm: Container with posterior means, standard deviations, and model parameters.
    
    """

    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Data
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MDN(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        n_gaussians=n_gaussians,
        n_layers=n_layers,
    )
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi, mu, log_sigma = model(inputs)
            loss = mdn_spike_loss_with_varying_noise(
                pi,
                mu,
                log_sigma,
                targets,
                noise_std,
                penalty=penalty,
                beta_prior=beta_prior,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % print_every == 0:
            print(f"[Spiked-EMDN] Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Predict (all data)
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for X_batch, _, _ in full_loader:
            pi_pred, mu_pred, log_sigma_pred = model(X_batch)

    # Build full mixture params including the spike at 0 with prior sd=0
    mu_full = torch.cat([torch.zeros_like(mu_pred[:, :1]), mu_pred], dim=1)  # (N, K)
    sigma_full = torch.cat([torch.zeros_like(log_sigma_pred[:, :1]), torch.exp(log_sigma_pred)], dim=1)  # (N, K)

    pi_np = pi_pred.cpu().numpy()
    mu_np = mu_full.cpu().numpy()
    scale_np = sigma_full.sqrt().cpu().numpy()  # prior SDs (0 for spike)

    # Posterior moments per observation
    N = len(betahat)
    post_mean = torch.empty(N, dtype=torch.float32)
    post_mean2 = torch.empty(N, dtype=torch.float32)
    post_sd = torch.empty(N, dtype=torch.float32)

    for i in range(N):
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            location=torch.tensor(mu_np[i, :], dtype=torch.float32),
            scale=torch.tensor(scale_np[i, :], dtype=torch.float32),  # 0 for spike ⇒ total sd = se
        )
        result = posterior_mean_norm(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            log_pi=torch.log(torch.tensor(pi_np[i, :], dtype=torch.float32) + 1e-8),
            data_loglik=data_loglik,
            location=torch.tensor(mu_np[i, :], dtype=torch.float32),
            scale=torch.tensor(scale_np[i, :], dtype=torch.float32),
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    return EmdnPosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        location=mu_np,
        pi_np=pi_np,
        scale=scale_np,
        loss=running_loss,
        model_param=model.state_dict(),
    )
