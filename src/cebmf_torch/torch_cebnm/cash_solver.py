# Define dataset class that includes observation noise


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.torch_distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.torch_posterior import posterior_mean_norm

# Import utils.py directly
from cebmf_torch.utils.torch_utils_mix import autoselect_scales_mix_norm


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
        super(CashNet, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input layer
        x = self.relu(self.input_layer(x))

        # Hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        # Output layer
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
        penalized_log_likelihood_value = first_sum + (penalty - 1) * torch.log(
            pi_clamped
        )
    else:
        penalized_log_likelihood_value = first_sum

    return -penalized_log_likelihood_value


# Class to store the results
class cash_PosteriorMeanNorm:
    def __init__(
        self, post_mean, post_mean2, post_sd, pi_np, scale, loss=0, model_param=None
    ):
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
    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale = autoselect_scales_mix_norm(
        betahat=betahat, sebetahat=sebetahat, max_class=num_classes
    )
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
    # Training loop
    for epoch in range(n_epochs):
        total_cash_loss = 0

        for inputs, targets, noise_std in dataloader:
            batch_loglik = get_data_loglik_normal_torch(
                betahat=targets, sebetahat=noise_std, location=0 * scale, scale=scale
            )
            optimizer_cash.zero_grad()
            outputs = model_cash(inputs)

            cash_loss = pen_loglik_loss(
                pred_pi=outputs, marginal_log_lik=batch_loglik, penalty=penalty
            )
            cash_loss.backward()
            optimizer_cash.step()
            total_cash_loss += cash_loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs},   Variance Loss: {total_cash_loss / len(dataloader):.4f}"
            )
            # After training the model, compute the posterior mean

    model_cash.eval()

    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for X_batch, _, _ in train_loader:
        all_pi_values = model_cash(X_batch)

    # Initialize arrays to store the results
    J = len(betahat)
    post_mean = torch.empty(J, dtype=torch.float32)
    post_mean2 = torch.empty(J, dtype=torch.float32)
    post_sd = torch.empty(J, dtype=torch.float32)
    data_loglik = get_data_loglik_normal_torch(
        betahat=betahat, sebetahat=sebetahat, location=0 * scale, scale=scale
    )
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
