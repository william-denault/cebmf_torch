from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class Prior:
    """
    Result from fitting a prior distribution.

    Attributes
    ----------
    post_mean : torch.Tensor
        Posterior means for each observation.
    post_mean2 : torch.Tensor
        Posterior second moments for each observation.
    loss : float
        Final training loss or log-likelihood.
    model_param : Any or None, optional
        Trained model parameters (state_dict) or other metadata.
    pi0_null : torch.Tensor, float, or None, optional
        Null component probability (if applicable).
    pi_slab : torch.Tensor, float, or None, optional
        Slab component probability (if applicable).
    """

    post_mean: Tensor
    post_mean2: Tensor
    loss: float
    model_param: Any | None = None
    pi0_null: Tensor | float | None = None
    pi_slab: Tensor | float | None = None


class PriorBuilder(ABC):
    """
    Base class for all prior distributions.
    """

    @abstractmethod
    def fit(
        self,
        X: Tensor | None,
        betahat: Tensor,
        sebetahat: Tensor,
        model_param: Any | None = None,
    ) -> Prior:
        """
        Fit the prior and return posterior estimates.

        Parameters
        ----------
        X : torch.Tensor or None
            Covariate matrix (can be None for non-covariate priors).
        betahat : torch.Tensor
            Effect size estimates.
        sebetahat : torch.Tensor
            Standard errors of effect sizes.
        model_param : Any or None, optional
            Additional model parameters (for warm starts).

        Returns
        -------
        Prior
            Result object with posterior means, second moments, and metadata.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a string identifier for this prior.

        Returns
        -------
        str
            String identifier for the prior.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
