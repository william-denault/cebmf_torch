from enum import StrEnum, auto
from typing import Any

from torch import Tensor

from cebmf_torch.ebnm.ash import ash

from .base import Prior, PriorBuilder


class MixturePriorType(StrEnum):
    NORM = auto()
    EXP = auto()


class ASHBuilder(PriorBuilder):
    def __init__(self, type: MixturePriorType):
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
        obj = ash(betahat, sebetahat, prior=str(self.type))
        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-float(obj.log_lik),
            model_param=model_param,
            pi0_null=obj.pi0,
        )
