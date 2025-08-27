import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Callable, Dict

from .torch_ash import ash
from .torch_utils_mix import autoselect_scales_mix_norm, autoselect_scales_mix_exp
from .torch_device import get_device
from .torch_mix_opt import optimize_pi_logL
from .priors_torch import get_prior_function_torch, PriorResultTorch

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
                    prior_L: str | Callable = "norm",
                    prior_F: str | Callable = "norm",
                    type_noise: str ='constant',
                          device: Optional[torch.device] = None,
                    allow_backfitting: bool = True,
                    prune_pi0 : float= 1- 1e-3):
        self.device = device or get_device()
        self.Y = data.to(self.device).float()
        self.mask = (~torch.isnan(self.Y)).float()   # 1 where observed, 0 where NaN
        self.Y0 = torch.nan_to_num(self.Y, nan=0.0)  # zeros where missing
        self.K = K
        self.N, self.P = self.Y.shape
        self.prior_L_fn = get_prior_function_torch(prior_L)  # string or callable -> callable
        self.prior_F_fn = get_prior_function_torch(prior_F)
        self.model_state_L = [None] * K
        self.model_state_F = [None] * K
        self.type_noise =type_noise
        self.L    = torch.zeros(self.N, K, device=self.device)
        self.L2   = torch.zeros(self.N, K, device=self.device)
        self.F    = torch.zeros(self.P, K, device=self.device)
        self.F2   = torch.zeros(self.P, K, device=self.device)
        self.tau  = torch.tensor(1.0, device=self.device)  # precision (1/var) 
        self.kl_l = torch.zeros(self.K, device=self.device)
        self.kl_f = torch.zeros(self. K, device=self.device)
        self.prune_thresh = prune_pi0
        self.pi0_L = [None] * K  # store latest pi0 for L[:,k]; scalar or Tensor or None
        self.pi0_F = [None] * K 
        self.obj = []
        self.allow_backfitting= allow_backfitting

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
    @torch.no_grad()
    def iter_once(self):
        eps = 1e-12
        # ensure 1-D KL holders (length K)
        if not hasattr(self, "kl_l"): self.kl_l = torch.zeros(self.K, device=self.device)
        if not hasattr(self, "kl_f"): self.kl_f = torch.zeros(self.K, device=self.device)


    # choose tau map depending on mode
        tau_map = None
        if self.type_noise == "constant":
            tau_scalar = float(self.tau.item())
        elif self.type_noise == "row_wise":
            tau_map = self.tau_map  # (N,P)
        elif self.type_noise == "column_wise":
            tau_map = self.tau_map  # (N,P)
        else:
            raise ValueError("Invalid type_noise")

        for k in range(self.K):
            self.update_factor(k, tau_map=tau_map, eps=eps)

        if self.allow_backfitting and self.K > 1:
            to_drop = [k for k in range(self.K) if self._should_prune_factor(k, self.prune_thresh)]
            if to_drop:
        # drop highest indices first to avoid reindex churn
                to_drop_sorted = sorted(to_drop, reverse=True)
                self._prune_indices(to_drop_sorted)
        self.update_tau()
        self.cal_obj()


    @torch.no_grad
    def update_factor(self, k: int, tau_map: Optional[Tensor] = None, eps: float = 1e-12) -> None:
        """
        Update L[:,k], F[:,k] and their second moments using the current priors.
        Handles scalar tau (tau_map=None) or elementwise tau (tau_map is (N,P)).
        """
        # ---------- Update L[:, k] ----------
        Rk = self._partial_residual_masked(k)            # (N,P), zeros where missing
        fk  = self.F[:, k]                                # (P,)
        fk2 = self.F2[:, k]                               # (P,)

        if tau_map is None:
            denom_l = (fk2.view(1, -1) * self.mask).sum(dim=1).clamp_min(eps)     # (N,)
            num_l   = (Rk @ fk)                                                   # (N,)
            se_l    = torch.sqrt(1.0 / (self.tau * denom_l))
        else:
            denom_l = (tau_map * (fk2.view(1, -1) * self.mask)).sum(dim=1).clamp_min(eps)  # (N,)
            num_l   = (tau_map * Rk) @ fk                                                 # (N,)
            se_l    = torch.sqrt(1.0 / denom_l)

        lhat = num_l / denom_l
       # print(denom_l)
        resL = self.prior_L_fn(
            X=getattr(self, "X_l", None),
            betahat=lhat,
            sebetahat=se_l,
            model_param=self.model_state_L[k]
        )
        self.model_state_L[k] = resL.model_param
        self.L[:, k]  = resL.post_mean
        self.L2[:, k] = resL.post_mean2
        self.kl_l[k]  = torch.as_tensor(resL.loss, device=self.device)
        self.pi0_L[k] = resL.pi0_null if hasattr(resL, "pi0_null") else None


    # ---------- Update F[:, k] ----------
        Rk = self._partial_residual_masked(k)            # recompute with updated L
        lk  = self.L[:, k]                               # (N,)
        lk2 = self.L2[:, k]                              # (N,)

        if tau_map is None:
            denom_f = (lk2.view(-1, 1) * self.mask).sum(dim=0).clamp_min(eps)      # (P,)
            num_f   = (Rk.T @ lk)                                                  # (P,)
            se_f    = torch.sqrt(1.0 / (self.tau * denom_f))
        else:
            denom_f = (tau_map * (lk2.view(-1, 1) * self.mask)).sum(dim=0).clamp_min(eps)  # (P,)
            num_f   = (tau_map * Rk).T @ lk                                                # (P,)
            se_f    = torch.sqrt(1.0 / denom_f)

        fhat = num_f / denom_f

        resF = self.prior_F_fn(
                    X=getattr(self, "X_f", None),
                    betahat=fhat,
                    sebetahat=se_f,
                    model_param=self.model_state_F[k]
                    )
        self.model_state_F[k] = resF.model_param
        self.F[:, k]  = resF.post_mean
        self.F2[:, k] = resF.post_mean2
# store as scalar on device; PriorResult.loss already = -log_lik
        self.kl_f[k]  = torch.as_tensor(resF.loss, device=self.device)
        self.pi0_F[k] = resF.pi0_null  if hasattr(resF, "pi0_null") else None

    def cal_obj(self):
    # Data term
        R = self.Y0 - self.L @ self.F.T
        R2 = (R * R) * self.mask

        if self.type_noise == "constant":
            m = self.mask.sum().clamp_min(1.0)
            ll = -0.5 * (
                m * (torch.log(torch.tensor(2*torch.pi, device=self.device)) - torch.log(self.tau))
                + self.tau * R2.sum()
                )
        else:
            tau_map = self.tau_map
            obs = self.mask.bool()
            ll = -0.5 * (
                torch.log(torch.tensor(2*torch.pi, device=self.device)) * obs.sum()
                - torch.log(tau_map[obs]).sum()
                + (tau_map * R2)[obs].sum()
                )

        # KL term from priors; PriorResult.loss = -log_lik (positive KL-like)
        KL = self.kl_l.sum() + self.kl_f.sum()

    # Final objective matches NumPy convention: obj = ll - KL
        obj = (ll - KL).item()
        self.obj.append(obj)


        
    @torch.no_grad()
    def update_fitted_value(self):
        self.Y_fit = self.L @ self.F.T

    @torch.no_grad()    
    def fit(self, maxit: int = 50):
        self.initialize()
        for _ in range(maxit):
             self.iter_once()
        return CEBMFResult(self.L, self.F, self.tau, self.obj)

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
    

    def _pi0_min_value(self, pi0_val) -> float:
        if pi0_val is None:
            return float("-inf")
        if isinstance(pi0_val, torch.Tensor):
            if pi0_val.numel() == 0:
                return float("-inf")
            return float(pi0_val.min().item())
        return float(pi0_val)

    def _should_prune_factor(self, k: int, thresh: float) -> bool:
        """
        Remove factor k if we have π₀ info and the smallest π₀ across coordinates
        (for any side that provided it) is ≥ thresh. (Your spec: use the *lowest* π₀.)
        """
        pi0_min_L = self._pi0_min_value(self.pi0_L[k])
        pi0_min_F = self._pi0_min_value(self.pi0_F[k])
        # if neither side provided π0, don't prune
        if pi0_min_L == float("-inf") and pi0_min_F == float("-inf"):
            return False
        # If either side indicates "all near spike", prune.
        return (pi0_min_L >= thresh) or (pi0_min_F >= thresh)

    def _prune_indices(self, idxs: list[int]) -> None:
        """In-place prune of K and all factor-aligned structures."""
        if not idxs:
            return
        keep = [i for i in range(self.K) if i not in idxs]
        self.L  = self.L[:, keep]
        self.L2 = self.L2[:, keep]
        self.F  = self.F[:, keep]
        self.F2 = self.F2[:, keep]
        self.kl_l = self.kl_l[keep]
        self.kl_f = self.kl_f[keep]
        self.model_state_L = [self.model_state_L[i] for i in keep]
        self.model_state_F = [self.model_state_F[i] for i in keep]
        self.pi0_L = [self.pi0_L[i] for i in keep]
        self.pi0_F = [self.pi0_F[i] for i in keep]
        self.K = len(keep)



 

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


 