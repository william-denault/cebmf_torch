from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

from torch import Tensor

from cebmf_torch.cebnm.cash_solver import cash_posterior_means
from cebmf_torch.cebnm.cov_gb_prior import cgb_posterior_means
from cebmf_torch.cebnm.cov_sharp_gb_prior import sharp_cgb_posterior_means
from cebmf_torch.cebnm.emdn import emdn_posterior_means

from .base import Prior, PriorBuilder


class LearnedPriorType(StrEnum):
    CASH = auto()
    CGB = auto()
    CGB_SHARP = auto()
    EMDN = auto()


builder_functions: dict[LearnedPriorType, Callable] = {
    LearnedPriorType.CASH: cash_posterior_means,
    LearnedPriorType.CGB: cgb_posterior_means,
    LearnedPriorType.CGB_SHARP: sharp_cgb_posterior_means,
    LearnedPriorType.EMDN: emdn_posterior_means,
}


class LearnedBuilder(PriorBuilder):
    def __init__(self, type: LearnedPriorType):
        self.type = type

    @property
    def name(self) -> str:
        return str(self.type)

    def fit(
        self,
        X: Tensor | None,
        betahat: Tensor,
        sebetahat: Tensor,
        model_param: Any | None = None,
    ) -> Prior:
        obj = builder_functions[self.type](
            X, betahat, sebetahat, model_param=model_param
        )

        # A bit annoying that the different types have different ways of handling pi0
        match self.type:
            case LearnedPriorType.CASH:
                # optional: could expose from obj.pi_np
                pi0_null = obj.pi_np[:, 0]
            case LearnedPriorType.CGB | LearnedPriorType.CGB_SHARP:
                # π₀(x) from the covariate model
                pi0_null = obj.pi
            case LearnedPriorType.EMDN:
                pi0_null = None
            case _:
                raise ValueError(f"Unknown prior type: {self.type}")

        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-float(obj.log_lik),
            model_param=model_param,
            pi0_null=pi0_null,
        )
