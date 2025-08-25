import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Callable, Dict

from .torch_ash import ash
from .torch_utils_mix import autoselect_scales_mix_norm, autoselect_scales_mix_exp
from .torch_device import get_device
from .torch_mix_opt import optimize_pi_logL

@dataclass
class CEBMFResult:
    L: Tensor
    F: Tensor
    tau: Tensor
    history_obj: list

class cEBMF:
    """
    Pure-PyTorch EBMF with proper NaN handling:
    - Observed-mask weighting in lhat/fhat and their standard errors.
    - Constant noise precision (scalar tau).
    - Mini-batch optimization for mixture weights inside ash().
    """
    def __init__(self, data: Tensor, K: int = 5, prior_L: str = "norm", prior_F: str = "norm", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.Y = data.to(self.device).float()
        self.mask = (~torch.isnan(self.Y)).float()   # 1 where observed, 0 where NaN
        self.Y0 = torch.nan_to_num(self.Y, nan=0.0)  # zeros where missing

        self.K = K
        self.N, self.P = self.Y.shape
        self.prior_L = prior_L
        self.prior_F = prior_F

        self.L = torch.zeros(self.N, K, device=self.device)
        self.F = torch.zeros(self.P, K, device=self.device)
        self.tau = torch.tensor(1.0, device=self.device)  # precision (1/var)

        self.obj = []

    def _impute_nan(self, Y: Tensor) -> Tensor:
        # Column-mean imputation in pure torch (for SVD init only)
        mask = ~torch.isnan(Y)
        if not mask.any():
            return Y
        col_means = torch.nanmean(Y, dim=0)
        Y_imp = Y.clone()
        idx = torch.where(~mask)
        Y_imp[idx] = col_means[idx[1]]
        return Y_imp

    def initialize(self, method: str = "svd"):
        Y_for_init = self._impute_nan(self.Y)
        if method == "svd":
            U, S, Vh = torch.linalg.svd(Y_for_init, full_matrices=False)
            K = min(self.K, S.shape[0])
            self.L = (U[:, :K] * S[:K]).contiguous()
            self.F = Vh[:K, :].T.contiguous()
        else:
            # random init
            self.L = torch.randn(self.N, self.K, device=self.device) * 0.01
            self.F = torch.randn(self.P, self.K, device=self.device) * 0.01
        self.update_tau()

    def update_tau(self):
        # Use only observed entries
        R = self.Y0 - self.L @ self.F.T
        sse = ((R * R) * self.mask).sum()
        m = self.mask.sum().clamp_min(1.0)
        self.tau = 1.0 / (sse / m + 1e-12)

    def _partial_residual_masked(self, k: int) -> Tensor:
        # Rk for observed entries only
        recon = self.L @ self.F.T
        k_contrib = torch.outer(self.L[:, k], self.F[:, k])
        Rk = (self.Y0 - (recon - k_contrib)) * self.mask
        return Rk

    def iter_once(self):
        eps = 1e-12
        for k in range(self.K):
            # --- Update L[:, k] using observed mask ---
            Rk = self._partial_residual_masked(k)           # (N,P), zeros where missing
            fk = self.F[:, k]                                # (P,)
            fk2 = fk * fk                                    # (P,)
            # denom per row i: sum_j fk^2 * mask_ij
            denom_l = (fk2.view(1, -1) * self.mask).sum(dim=1).clamp_min(eps)  # (N,)
            # numerator per row i: sum_j Rk_ij * fk_j
            num_l = (Rk @ fk)                                # (N,)
            lhat = num_l / denom_l
            se_l = torch.sqrt(1.0 / (self.tau * denom_l))

            if self.prior_L == "norm":
                resL = ash(lhat, se_l, prior="norm" )
            elif self.prior_L == "exp":
                resL = ash(lhat, se_l, prior="exp" )
            else:
                raise ValueError("prior_L must be 'norm' or 'exp'")
            self.L[:, k] = resL.post_mean

            # --- Update F[:, k] using observed mask ---
            Rk = self._partial_residual_masked(k)            # recompute with updated L
            lk = self.L[:, k]                                # (N,)
            lk2 = lk * lk                                    # (N,)
            # denom per col j: sum_i lk^2 * mask_ij
            denom_f = (lk2.view(-1, 1) * self.mask).sum(dim=0).clamp_min(eps)  # (P,)
            num_f = (Rk.T @ lk)                              # (P,)
            fhat = num_f / denom_f
            se_f = torch.sqrt(1.0 / (self.tau * denom_f))

            if self.prior_F == "norm":
                resF = ash(fhat, se_f, prior="norm" )
            elif self.prior_F == "exp":
                resF = ash(fhat, se_f, prior="exp")
            else:
                raise ValueError("prior_F must be 'norm' or 'exp'")
            self.F[:, k] = resF.post_mean

        self.update_tau()
        self.cal_obj()

    def cal_obj(self):
        # Gaussian data likelihood on observed entries only
        R = self.Y0 - self.L @ self.F.T
        sse = ((R * R) * self.mask).sum()
        m = self.mask.sum().clamp_min(1.0)
        ll = -0.5 * ( m * (torch.log(2*torch.tensor(torch.pi, device=self.device)) - torch.log(self.tau)) + self.tau * sse )
        self.obj.append(ll.item())

    def fit(self, maxit: int = 50):
        self.initialize()
        for _ in range(maxit):
            self.iter_once()
        return CEBMFResult(self.L, self.F, self.tau, self.obj)
    
    def get_fitted_value(self):
        self.Y_fit = self.L @ self.F.T