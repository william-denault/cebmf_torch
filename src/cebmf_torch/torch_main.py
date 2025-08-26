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
    def __init__(self,
                    data: Tensor,
                    K: int = 5,
                    prior_L: str = "norm",
                    prior_F: str = "norm",
                    type_noise: str ='constant',
                          device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.Y = data.to(self.device).float()
        self.mask = (~torch.isnan(self.Y)).float()   # 1 where observed, 0 where NaN
        self.Y0 = torch.nan_to_num(self.Y, nan=0.0)  # zeros where missing
        self.K = K
        self.N, self.P = self.Y.shape
        self.prior_L = prior_L
        self.prior_F = prior_F
        self.type_noise =type_noise
        self.L = torch.zeros(self.N, K, device=self.device)
        self.L2 = torch.zeros(self.N, K, device=self.device)
        self.F = torch.zeros(self.P, K, device=self.device)
        self.F2 = torch.zeros(self.P, K, device=self.device)
        self.tau = torch.tensor(1.0, device=self.device)  # precision (1/var)
        self.L  = torch.zeros(self.N, K, device=self.device)
        self.kl_l = torch.zeros(self.N, K, device=self.device)
        self.kl_f = torch.zeros(self.P, K, device=self.device)
 

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
        self.L2 = self.L*self.L
        self.F2 = self.F*self.F
        self.update_tau()
    @torch.no_grad()
    @torch.no_grad()
    def update_tau(self):
        """
        Matches NumPy behavior:
        - 'constant'   -> scalar tau; also provides tau_map (N,P) if you need it
        - 'row_wise'   -> tau_row (N,), tau_map broadcast to (N,P)
        - 'column_wise'-> tau_col (P,), tau_map broadcast to (N,P)
        """
        eps = 1e-12
        R2 = self.expected_residuals_squared()        # (N,P), zeros at missing
        N, P = self.N, self.P

        if self.type_noise == 'constant':
            m = self.mask.sum().clamp_min(1.0)
            mean_R2 = (R2.sum() / m)
            tau = 1.0 / (mean_R2 + eps)
            self.tau = tau                             # scalar (back-compat)
        # If you need a full map like NumPy's np.full(...):
            self.tau_map = torch.full((N, P), tau.item(), device=self.device, dtype=R2.dtype)

        elif self.type_noise == 'row_wise':
            m_row = self.mask.sum(dim=1).clamp_min(1.0)            # (N,)
            mean_R2_row = (R2.sum(dim=1) / m_row)                  # (N,)
            tau_row = 1.0 / (mean_R2_row + eps)                    # (N,)
            self.tau_row = tau_row
            self.tau_map = tau_row.view(-1, 1).expand(N, P)        # (N,P)
            self.tau = self.tau_map                                # if downstream expects elementwise

        elif self.type_noise == 'column_wise':
            m_col = self.mask.sum(dim=0).clamp_min(1.0)            # (P,)
            mean_R2_col = (R2.sum(dim=0) / m_col)                  # (P,)
            tau_col = 1.0 / (mean_R2_col + eps)                    # (P,)
            self.tau_col = tau_col
            self.tau_map = tau_col.view(1, -1).expand(N, P)        # (N,P)
            self.tau = self.tau_map                                # if downstream expects elementwise

        else:
            raise ValueError("type_noise must be 'constant', 'row_wise', or 'column_wise'")

    @torch.no_grad()
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
            fk2 = self.F2[:, k]                                   # (P,)
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
            self.L2[:, k] = resL.post_mean2

            # --- Update F[:, k] using observed mask ---
            Rk = self._partial_residual_masked(k)            # recompute with updated L
            lk = self.L[:, k]                                # (N,)
            lk2 = self.L2[:, k]                                   # (N,)
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
            self.F2[:, k] = resF.post_mean2
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
    
    @torch.no_grad()
    def update_fitted_value(self):
        self.Y_fit = self.L @ self.F.T
        
 
    

    @torch.no_grad()
    def expected_residuals_squared(self):
        """
        E[(Y - sum_k L_k F_k)^2] on observed entries.
        Uses: (Y - E[Y])^2 - sum_k (E[L]^2)(E[F]^2)^T + sum_k E[L^2] E[F^2]^T
        """
        eps = 1e-12
        Yfit = self.L @ self.F.T                                  # (N,P)
        resid_mean_sq = (self.Y0 - Yfit).pow(2)                   # (N,P)
        first_moment_sq = (self.L.pow(2)) @ (self.F.pow(2)).T     # Σ_k (E[L]^2)(E[F]^2)^T
        second_moment   = self.L2 @ self.F2.T                     # Σ_k E[L^2] E[F^2]^T
        R2 = resid_mean_sq - first_moment_sq + second_moment
        R2 = (R2 * self.mask).clamp_min(0.0)                      # zero where missing; no negatives
        return R2


 

def normal_means_loglik(x: torch.Tensor,
                        s: torch.Tensor,
                        Et: torch.Tensor,
                        Et2: torch.Tensor) -> torch.Tensor:
    """
    Normal means log-likelihood in PyTorch.

    Args:
        x   : observed data (Tensor)
        s   : standard errors (Tensor, must be >0 and finite)
        Et  : posterior means (Tensor)
        Et2 : posterior second moments (Tensor)

    Returns:
        torch scalar (log-likelihood)
    """
    eps = 1e-12
    # mask valid entries: finite and positive s
    mask = torch.isfinite(s) & (s > 0)

    if not mask.any():
        return torch.tensor(float("nan"), device=x.device)

    x = x[mask]
    s = s[mask]
    Et = Et[mask]
    Et2 = Et2[mask]

    term = Et2 - 2 * x * Et + x**2
    loglik = -0.5 * torch.sum(torch.log(2 * torch.pi * s**2 + eps) + term / (s**2 + eps))
    return loglik
