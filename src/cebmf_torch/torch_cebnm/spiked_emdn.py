import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
# Mixture Density Network with spike at 0
# -------------------------
class MDN(nn.Module):
    """
    n_components = K + 1 (component 0 is the spike at 0)
    Outputs:
      pi:   (N, K+1)    mixture weights
      mu:   (N, K+1)    means, mu[:,0] == 0
      sigma:(N, K+1)    prior stds, sigma[:,0] == 0 (spike prior)
    """
    def __init__(self, input_dim, hidden_dim, n_components, n_layers=4):
        super().__init__()
        assert n_components >= 2, "n_components must be >= 2 (at least one spike + one slab)."
        self.K = n_components - 1  # number of slabs

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Heads
        self.pi_head = nn.Linear(hidden_dim, n_components)       # spike + slabs
        self.mu_slab_head = nn.Linear(hidden_dim, self.K)        # slabs only
        self.log_sigma_slab_head = nn.Linear(hidden_dim, self.K) # slabs only

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))

        # Weights across spike+slabs
        pi = torch.softmax(self.pi_head(h), dim=1)               # (N, K+1)

        # Slab parameters
        mu_slab = self.mu_slab_head(h)                           # (N, K)
        sigma_slab = torch.exp(self.log_sigma_slab_head(h))      # (N, K)

        # Prepend spike columns
        N = x.shape[0]
        device, dtype = x.device, mu_slab.dtype
        zero_col = torch.zeros((N, 1), device=device, dtype=dtype)

        mu = torch.cat([zero_col, mu_slab], dim=1)               # (N, K+1)
        sigma = torch.cat([zero_col, sigma_slab], dim=1)         # (N, K+1)

        return pi, mu, sigma


# -------------------------
# Loss (spike handled via sigma[:,0]==0) + optional L2 on slabs
# -------------------------
def mdn_spike_loss_with_varying_noise(pi, mu, sigma, betahat, sebetahat, penalty=0.0, eps=1e-8):
    """
    betahat:   (N,)
    sebetahat: (N,)
    pi:        (N, K+1)
    mu:        (N, K+1) with mu[:,0]==0
    sigma:     (N, K+1) with sigma[:,0]==0
    """
    eps = 1e-12
    b = betahat.unsqueeze(1)                    # (N,1)
    se = sebetahat.unsqueeze(1)                 # (N,1)

    # Predictive std: sqrt(se^2 + sigma^2). Spike gets total_sigma = se.
    total_sigma = torch.sqrt(se**2 + sigma**2)  # (N, K+1)
    dist = torch.distributions.Normal(loc=mu, scale=total_sigma)

    log_probs = dist.log_prob(b.expand_as(mu)) + torch.log(pi.clamp_min(eps))
    nll = -torch.logsumexp(log_probs, dim=1).mean()

    if penalty > 1:
        # L2 on slab means (skip spike col 0)
        #nll = nll + penalty * (mu[:, 1:]**2).mean()
        pi0_clamped = pi[:,0].mean().clamp_min(eps)
        penalty_term = (penalty - 1.0) *  (pi0_clamped)
        nll = nll - penalty_term
    return nll


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
        loss=0.0,
        model_param=None,
    ):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.location = location     # mu (N, K+1)
        self.pi_np = pi_np           # pi (N, K+1)
        self.scale = scale           # sigma (N, K+1)
        self.loss = loss
        self.model_param = model_param


# -------------------------
# Main solver (pure Torch; no sklearn)
# -------------------------
def spiked_emdn_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=50,
    n_layers=4,
    n_gaussians=5,     # total components incl. spike
    hidden_dim=64,
    batch_size=512,
    lr=1e-3,
    penalty=0.0,     # L2 strength on slab means
    model_param=None,
    verbose_every=10,
):
    # ---- Standardize X with Torch (avoid sklearn) ----
    X = torch.as_tensor(X, dtype=torch.float32)
    if X.ndim == 1:
        X = X.view(-1, 1)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std  = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    X_scaled = (X - X_mean) / X_std

    betahat = torch.as_tensor(betahat, dtype=torch.float32)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32)

    # Dataset + DataLoader
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MDN(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        n_components=n_gaussians,
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
            pi, mu, sigma = model(inputs)
            loss = mdn_spike_loss_with_varying_noise(
                pi, mu, sigma, targets, noise_std, penalty=penalty
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"[EMDN] Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Predict on full data
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for X_batch, _, _ in full_loader:
            pi, mu, sigma = model(X_batch)

    # Posterior means per observation via your utilities
    from cebmf_torch.utils.torch_distribution_operation import get_data_loglik_normal_torch
    from cebmf_torch.utils.torch_posterior import posterior_mean_norm

    J = betahat.shape[0]
    post_mean = torch.empty(J, dtype=torch.float32)
    post_mean2 = torch.empty(J, dtype=torch.float32)
    post_sd = torch.empty(J, dtype=torch.float32)

    for i in range(J):
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat[i:i+1],
            sebetahat=sebetahat[i:i+1],
            location=mu[i, :],
            scale=sigma[i, :],             # spike has 0 prior sd
        )
        result = posterior_mean_norm(
            betahat=betahat[i:i+1],
            sebetahat=sebetahat[i:i+1],
            log_pi=torch.log(pi[i, :].clamp_min(1e-12)),
            data_loglik=data_loglik,
            location=mu[i, :],
            scale=sigma[i, :],
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    return EmdnPosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        location=mu,
        pi_np=pi,
        scale=sigma,
        loss=running_loss,
        model_param=model.state_dict(),
    )
