# Define dataset class that includes observation noise


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch

# Import utils.py directly
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm
from cebmf_torch.utils.posterior import posterior_mean_norm


# Define dataset class that includes observation noise
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.betahat = torch.as_tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.betahat[idx],
            self.sebetahat[idx],
        )  # Return the noise_std (sebetahat) as well


# Define the MeanNet model


# Define the CashNet model
class CashNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers):
        """
        Initialize a neural network for CASH (Covariate Adaptive Shrinkage).

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of hidden units in each layer.
        num_classes : int
            Number of mixture components (output classes).
        n_layers : int
            Number of hidden layers.
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the CASH network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            Mixture weights for each observation, shape (N, num_classes).
        """
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x


# Custom loss function
def pen_loglik_loss(pred_pi, marginal_log_lik, penalty=1.1, epsilon=1e-10):
    L_batch = torch.exp(marginal_log_lik)
    inner_sum = torch.sum(pred_pi * L_batch, dim=1)
    inner_sum = torch.clamp(inner_sum, min=epsilon)
    first_sum = torch.sum(torch.log(inner_sum))

    if penalty > 1:
        pi_clamped = torch.clamp(torch.sum(pred_pi[:, 0]), min=epsilon)
        penalized_log_likelihood_value = first_sum + (penalty - 1) * torch.log(pi_clamped)
    else:
        penalized_log_likelihood_value = first_sum

    return -penalized_log_likelihood_value


# Class to store the results
class cash_PosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd, pi_np, scale, loss=0, model_param=None):
        """
        Container for the results of the CASH posterior mean estimation.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means for each observation.
        post_mean2 : torch.Tensor
            Posterior second moments for each observation.
        post_sd : torch.Tensor
            Posterior standard deviations for each observation.
        pi_np : torch.Tensor
            Mixture weights for each observation.
        scale : torch.Tensor
            Mixture component scales.
        loss : float, optional
            Final training loss or log-likelihood.
        model_param : dict, optional
            Trained model parameters (state_dict).
        """
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi_np = pi_np
        self.loss = loss
        self.scale = scale
        self.model_param = model_param


# Main function to train the model and compute posterior means, mean^2, and standard deviations
def cash_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=20,
    n_layers=4,
    num_classes=20,
    hidden_dim=64,
    batch_size=128,
    lr=0.001,
    model_param=None,
    penalty=1.5,
):
    """
    Fit a CASH (Covariate Adaptive Shrinkage) model and compute posterior means,
    second moments, and standard deviations.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Covariates for each observation, shape (n_samples, n_features).
    betahat : torch.Tensor or np.ndarray
        Observed effect estimates, shape (n_samples,).
    sebetahat : torch.Tensor or np.ndarray
        Standard errors of the effect estimates, shape (n_samples,).
    n_epochs : int, optional
        Number of training epochs (default=20).
    n_layers : int, optional
        Number of hidden layers in the neural network (default=4).
    num_classes : int, optional
        Number of mixture components (default=20).
    hidden_dim : int, optional
        Number of hidden units in each layer (default=64).
    batch_size : int, optional
        Batch size for training (default=128).
    lr : float, optional
        Learning rate for the optimizer (default=0.001).
    model_param : dict, optional
        Pre-trained model parameters to initialize the network.
    penalty : float, optional
        Penalty for spike probability (default=1.5).

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with posterior means, standard deviations, and model parameters.
    """
    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, max_class=num_classes)
    # Create dataset and dataloader
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = X_scaled.shape[1]
    model_cash = CashNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        n_layers=n_layers,
    )
    optimizer_cash = optim.Adam(model_cash.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        total_cash_loss = 0

        for inputs, targets, noise_std in dataloader:
            batch_loglik = get_data_loglik_normal_torch(
                betahat=targets, sebetahat=noise_std, location=0 * scale, scale=scale
            )
            optimizer_cash.zero_grad()
            outputs = model_cash(inputs)

            cash_loss = pen_loglik_loss(pred_pi=outputs, marginal_log_lik=batch_loglik, penalty=penalty)
            cash_loss.backward()
            optimizer_cash.step()
            total_cash_loss += cash_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs},   Variance Loss: {total_cash_loss / len(dataloader):.4f}")

    model_cash.eval()

    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for X_batch, _, _ in train_loader:
        all_pi_values = model_cash(X_batch)

    # Initialize arrays to store the results
    J = len(betahat)
    post_mean = torch.empty(J, dtype=torch.float32)
    post_mean2 = torch.empty(J, dtype=torch.float32)
    post_sd = torch.empty(J, dtype=torch.float32)
    data_loglik = get_data_loglik_normal_torch(betahat=betahat, sebetahat=sebetahat, location=0 * scale, scale=scale)
    # Estimate posterior means for each observation

    for i in range(len(betahat)):
        log_pi_i = torch.log(torch.clamp(all_pi_values[i, :], min=1e-300))
        result = posterior_mean_norm(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            log_pi=log_pi_i,
            data_loglik=data_loglik[i, :],
            location=[0],
            scale=scale,  # Assuming this is available from earlier in your code
        )
        post_mean[i] = result.post_mean  # <-- take scalar
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    return cash_PosteriorMeanNorm(
        post_mean,
        post_mean2,
        post_sd,
        pi_np=all_pi_values,
        loss=total_cash_loss,
        scale=scale,
        model_param=model_param,
    )
