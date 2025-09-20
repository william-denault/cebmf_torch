import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm
from cebmf_torch.utils.posterior import posterior_mean_norm
from cebmf_torch.utils.standard_scaler import standard_scale


# ---- Dataset: assumes tensors already on correct device/dtype
class DensityRegressionDataset(Dataset):
    def __init__(self, X: torch.Tensor, betahat: torch.Tensor, sebetahat: torch.Tensor):
        self.X = X
        self.betahat = betahat
        self.sebetahat = sebetahat

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


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
        return self.softmax(self.output_layer(x))

# Custom loss function
def pen_loglik_loss(pred_pi, marginal_log_lik, penalty=1.5, epsilon=1e-10):
    L_batch = torch.exp(marginal_log_lik)                 # (B, K)
    inner_sum = torch.sum(pred_pi * L_batch, dim=1)       # (B,)
    inner_sum = torch.clamp(inner_sum, min=epsilon)
    first_sum = torch.sum(torch.log(inner_sum))           # scalar

    if penalty > 1:
        # penalize the (assumed) spike component's total mass in the batch
        pi_clamped = torch.clamp(torch.sum(pred_pi[:, 0]), min=epsilon)
        penalized_log_likelihood_value = first_sum + (penalty - 1) * torch.log(pi_clamped)
    else:
        penalized_log_likelihood_value = first_sum

    return -penalized_log_likelihood_value


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

# Class to store the results
 



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
    device: torch.device | None = None,
):
    """
    GPU-native CASH training and posterior computation.
     
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- to tensors on device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)

    # ---- standardize on device
    X_scaled = standard_scale(X)  # your function returns just scaled tensor

    # ---- mixture scales (ensure tensor on device)
    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, max_class=num_classes)
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    else:
        scale = scale.to(device=device, dtype=torch.float32)

    # ---- dataset / loader (CUDA tensors => num_workers must be 0)
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- model / optimizer on device
    input_dim = X_scaled.shape[1]
    model_cash = CashNet(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, n_layers=n_layers).to(device)
    if model_param is not None:
        model_cash.load_state_dict(model_param)
    optimizer_cash = optim.Adam(model_cash.parameters(), lr=lr)

    # ---- training
    total_cash_loss = 0.0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            # Compute (log)likelihood for this batch and current global scales
            batch_loglik = get_data_loglik_normal_torch(
                betahat=targets, sebetahat=noise_std, location=0 * scale, scale=scale
            )
            optimizer_cash.zero_grad()
            outputs = model_cash(inputs)
            cash_loss = pen_loglik_loss(pred_pi=outputs, marginal_log_lik=batch_loglik, penalty=penalty)
            cash_loss.backward()
            optimizer_cash.step()
            epoch_loss += cash_loss.item()

        total_cash_loss = epoch_loss
        if (epoch + 1) % 10 == 0:
            print(f"[CASH] Epoch {epoch + 1}/{n_epochs} | Loss: {epoch_loss / max(1, len(dataloader)):.4f}")

    # ---- full-batch inference (no grad)
    model_cash.eval()
    with torch.no_grad():
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        for X_batch, _, _ in train_loader:
            all_pi_values = model_cash(X_batch)  # (N, K)
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat, sebetahat=sebetahat, location=0 * scale, scale=scale
        )  # (N, K)

        # Allocate outputs on device
        J = betahat.shape[0]
        post_mean = torch.empty(J, dtype=torch.float32, device=device)
        post_mean2 = torch.empty(J, dtype=torch.float32, device=device)
        post_sd = torch.empty(J, dtype=torch.float32, device=device)

        # Per-observation posterior (kept as-is; can be vectorized later)
        eps = 1e-300
        for i in range(J):
            log_pi_i = torch.log(torch.clamp(all_pi_values[i, :], min=eps))
            res_i = posterior_mean_norm(
                betahat=betahat[i : i + 1],
                sebetahat=sebetahat[i : i + 1],
                log_pi=log_pi_i,
                data_loglik=data_loglik[i, :],
                location=[0],    # your routine expects this form
                scale=scale,
            )
            post_mean[i] = res_i.post_mean
            post_mean2[i] = res_i.post_mean2
            post_sd[i] = res_i.post_sd

    return cash_PosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_np=all_pi_values,      # (N, K) on device
        loss=total_cash_loss,
        scale=scale,              # (K,) on device
        model_param=model_cash.state_dict(),
    ) 
 