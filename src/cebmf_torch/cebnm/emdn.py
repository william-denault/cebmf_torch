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
# Mixture Density Network
# -------------------------
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians, n_layers=4):
        """
        Initialize a Mixture Density Network (MDN).

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of hidden units in each layer.
        n_gaussians : int
            Number of Gaussian components in the mixture.
        n_layers : int, optional
            Number of hidden layers (default is 4).
        """
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.pi = nn.Linear(hidden_dim, n_gaussians)
        self.mu = nn.Linear(hidden_dim, n_gaussians)
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians)

    def forward(self, x):
        """
        Forward pass through the MDN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        pi : torch.Tensor
            Mixture weights, shape (N, K).
        mu : torch.Tensor
            Means for each component, shape (N, K).
        log_sigma : torch.Tensor
            Log standard deviations for each component, shape (N, K).
        """
        x = torch.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        pi = torch.softmax(self.pi(x), dim=1)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        return pi, mu, log_sigma


# -------------------------
# Loss function
# -------------------------
def mdn_loss_with_varying_noise(pi, mu, log_sigma, betahat, sebetahat):
    """
    Compute the negative log-likelihood loss for a mixture density network with varying noise.

    Parameters
    ----------
    pi : torch.Tensor
        Mixture weights, shape (N, K).
    mu : torch.Tensor
        Means for each component, shape (N, K).
    log_sigma : torch.Tensor
        Log standard deviations for each component, shape (N, K).
    betahat : torch.Tensor
        Observed effect estimates, shape (N,).
    sebetahat : torch.Tensor
        Standard errors of the effect estimates, shape (N,).

    Returns
    -------
    torch.Tensor
        The computed loss (scalar).
    """
    sigma = torch.exp(log_sigma)
    total_sigma = torch.sqrt(sigma**2 + sebetahat.unsqueeze(1) ** 2)
    dist = torch.distributions.Normal(mu, total_sigma)
    log_probs = dist.log_prob(betahat.unsqueeze(1)) + torch.log(pi)
    return -torch.logsumexp(log_probs, dim=1).mean()


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
        """
        Container for the results of the EMDN posterior mean estimation.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means for each observation.
        post_mean2 : torch.Tensor
            Posterior second moments for each observation.
        post_sd : torch.Tensor
            Posterior standard deviations for each observation.
        location : np.ndarray or torch.Tensor
            Mixture component means for each observation.
        pi_np : np.ndarray or torch.Tensor
            Mixture weights for each observation.
        scale : np.ndarray or torch.Tensor
            Mixture component standard deviations for each observation.
        loss : float, optional
            Final training loss.
        model_param : dict, optional
            Trained model parameters (state_dict).
        """
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
def emdn_posterior_means(
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
):
    """
    Fit a Mixture Density Network (MDN) to estimate the prior distribution of effects.

    In the EBNM problem, we observe estimates `betahat` with standard errors `sebetahat` and want to estimate
    the prior distribution of the true effects. The prior is modeled as a mixture of Gaussians with parameters
    predicted by a neural network.

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
        Number of hidden layers in the neural network (default=4).
    n_gaussians : int, optional
        Number of Gaussian components in the mixture (default=5).
    hidden_dim : int, optional
        Number of hidden units in each layer (default=64).
    batch_size : int, optional
        Batch size for training (default=512).
    lr : float, optional
        Learning rate for the optimizer (default=1e-3).
    model_param : dict, optional
        Pre-trained model parameters to initialize the network.

    Returns
    -------
    EmdnPosteriorMeanNorm
        Container with posterior means, standard deviations, and model parameters.
    """

    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dataset + DataLoader
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Init model
    model = MDN(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        n_gaussians=n_gaussians,
        n_layers=n_layers,
    )
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi, mu, log_sigma = model(inputs)
            loss = mdn_loss_with_varying_noise(pi, mu, log_sigma, targets, noise_std)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[EMDN] Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Prediction for all data
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for X_batch, _, _ in full_loader:
            pi, mu, log_sigma = model(X_batch)

    # Posterior means per observation
    J = len(betahat)
    post_mean = torch.empty(J, dtype=torch.float32)
    post_mean2 = torch.empty(J, dtype=torch.float32)
    post_sd = torch.empty(J, dtype=torch.float32)

    for i in range(len(betahat)):
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            location=mu[i, :],
            scale=torch.exp(log_sigma)[i, :],
        )
        result = posterior_mean_norm(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            log_pi=torch.log(pi[i, :]),
            data_loglik=data_loglik,
            location=mu[i, :],
            scale=torch.exp(log_sigma)[i, :],
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
        scale=torch.exp(log_sigma),
        loss=running_loss,
        model_param=model.state_dict(),
    )
