from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

from torch import T, Tensor


@dataclass
class Prior:
    """Result from fitting a prior distribution."""

    post_mean: Tensor
    post_mean2: Tensor
    loss: float
    model_param: Any | None = None
    pi0_null: Tensor | float | None = None
    pi_slab: Tensor | float | None = None


class PriorBuilder(ABC):
    """Base class for all prior distributions."""

    @abstractmethod
    def fit(
        self,
        X: Tensor | None,
        betahat: Tensor,
        sebetahat: Tensor,
        model_param: Any | None = None,
    ) -> Prior:
        """Fit the prior and return posterior estimates.

        Args:
            X: Covariate matrix (can be None for non-covariate priors)
            betahat: Effect size estimates
            sebetahat: Standard errors of effect sizes
            model_param: Additional model parameters (for warm starts)

        Returns:
            PriorResult with posterior means, second moments, and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a string identifier for this prior."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
