import torch


def standard_scale(X: torch.Tensor) -> torch.Tensor:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    X_scaled : torch.Tensor, shape (n_samples, n_features)
        Standardized data.
    """
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True)  # population std like sklearn
    std = torch.where(std == 0, torch.ones_like(std), std)  # avoid division by zero
    return (X - mean) / std

 