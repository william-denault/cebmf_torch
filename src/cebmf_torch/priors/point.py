from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

from torch import Tensor

from cebmf_torch.ebnm.point_exp import ebnm_point_exp
from cebmf_torch.ebnm.point_laplace import ebnm_point_laplace

from .base import Prior, PriorBuilder


class PointPriorType(StrEnum):
    LAPLACE = auto()
    EXP = auto()


builder_functions: dict[PointPriorType, Callable] = {
    PointPriorType.LAPLACE: ebnm_point_laplace,
    PointPriorType.EXP: ebnm_point_exp,
}


class PointBuilder(PriorBuilder):
    def __init__(self, type: PointPriorType):
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
        obj = builder_functions[self.type](betahat, sebetahat)
        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-float(obj.log_lik),
            model_param=model_param,
            pi0_null=obj.pi0,
            pi_slab=1.0 - float(obj.pi0),
        )
