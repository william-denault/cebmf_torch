import math
from dataclasses import dataclass
from enum import StrEnum, auto

import torch
from torch import Tensor

from cebmf_torch.priors import PRIOR_REGISTRY
from cebmf_torch.utils.device import get_device


@dataclass
class CEBMFResult:
    L: Tensor
    F: Tensor
    tau: Tensor
    history_obj: list


class NoiseType(StrEnum):
    CONSTANT = auto()
    ROW_WISE = auto()
    COLUMN_WISE = auto()


def _impute_nan(Y: Tensor) -> Tensor:
    # Column-mean imputation in pure torch (for SVD init only)
    mask = ~torch.isnan(Y)
    if not mask.any():
        return Y
    col_means = torch.nanmean(Y, dim=0)
    Y_imp = Y.clone()
    idx = torch.where(~mask)
    Y_imp[idx] = col_means[idx[1]]
    return Y_imp


@dataclass
class cEBMF:
    """
    Pure-PyTorch EBMF with proper NaN handling:
    - Observed-mask weighting in lhat/fhat and their standard errors.
    - Constant noise precision (scalar tau).
    - Mini-batch optimization for mixture weights inside ash().
    """

    data: Tensor
    K: int = 5
    prior_L: str = "norm"
    prior_F: str = "norm"
    type_noise: NoiseType = NoiseType.CONSTANT
    device: torch.device | None = None
    allow_backfitting: bool = True
    prune_thresh: float = 1 - 1e-3
    X_l: Tensor | None = None
    X_f: Tensor | None = None
    self_row_cov: bool = False
    self_col_cov: bool = False

    def __post_init__(self):
        self._validate_inputs()
        self.device = self.device or get_device()
        self.N, self.P = self.Y.shape
        self._initialise_priors()
        self._initialise_tensors()

    def _validate_inputs(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if torch.isnan(self.data).all():
            raise ValueError("Data cannot be all NaN")
        # More validation...

    def _initialise_priors(self):
        self.prior_L_fn = PRIOR_REGISTRY.get_builder(self.prior_L)
        self.prior_F_fn = PRIOR_REGISTRY.get_builder(self.prior_F)
        self.model_state_L = [None] * self.K
        self.model_state_F = [None] * self.K

    def _initialise_tensors(self):
        self.Y = self.data.to(self.device).float()
        self.mask = (~torch.isnan(self.Y)).float()  # 1 where observed, 0 where NaN
        self.Y0 = torch.nan_to_num(self.Y, nan=0.0)  # zeros where missing
        self.L = torch.zeros(self.N, self.K, device=self.device)
        self.L2 = torch.zeros(self.N, self.K, device=self.device)
        self.F = torch.zeros(self.P, self.K, device=self.device)
        self.F2 = torch.zeros(self.P, self.K, device=self.device)
        self.tau = torch.tensor(1.0, device=self.device)  # precision (1/var)
        self.kl_l = torch.zeros(self.K, device=self.device)
        self.kl_f = torch.zeros(self.K, device=self.device)
        self.pi0_L = [None] * self.K  # store latest pi0 for L[:,k]; scalar or Tensor or None
        self.pi0_F = [None] * self.K
        self.obj = []

    def initialize(self, method: str = "svd"):
        Y_for_init = _impute_nan(self.Y)
        if method == "svd":
            U, S, Vh = torch.linalg.svd(Y_for_init, full_matrices=False)
            K = min(self.K, S.shape[0])
            self.L = (U[:, :K] * S[:K]).contiguous()
            self.F = Vh[:K, :].T.contiguous()

        else:
            # random init
            self.L = torch.randn(self.N, self.K, device=self.device) * 0.01
            self.F = torch.randn(self.P, self.K, device=self.device) * 0.01
        self.L2 = self.L * self.L
        self.F2 = self.F * self.F
        self.update_tau()

    @torch.no_grad()
    def update_tau(self):
        """
        Matches NumPy behavior:
        - 'constant'   -> scalar tau; also provides tau_map (N,P) if you need it
        - 'row_wise'   -> tau_row (N,), tau_map broadcast to (N,P)
        - 'column_wise'-> tau_col (P,), tau_map broadcast to (N,P)
        """
        R2 = self.expected_residuals_squared()  # (N,P), zeros at missing

        match self.type_noise:
            case NoiseType.CONSTANT:
                dim = None
            case NoiseType.ROW_WISE:
                dim = 1
            case NoiseType.COLUMN_WISE:
                dim = 0
            case _:
                raise ValueError("type_noise must be 'constant', 'row_wise', or 'column_wise'")

        self._set_tau(R2, dim=dim)

    @torch.no_grad()
    def _set_tau(self, R2: Tensor, dim: None | int) -> None:
        m = self.mask.sum(dim=dim).clamp_min(1.0)
        mean_R2 = R2.sum(dim=dim) / m
        tau = 1.0 / (mean_R2)
        if dim is None:
            self.tau = tau  # scalar (back-compat)
            self.tau_map = torch.full((self.N, self.P), tau.item(), device=self.device, dtype=R2.dtype)
        elif dim == 1:
            self.tau_map = tau.view(-1, 1).expand(self.N, self.P)  # (N,P)
            self.tau = self.tau_map  # if downstream expects elementwise
        elif dim == 0:
            self.tau_map = tau.view(1, -1).expand(self.N, self.P)  # (N,P)
            self.tau = self.tau_map  # if downstream expects elementwise
        else:
            raise ValueError("dim must be None, 0, or 1")

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
        if not hasattr(self, "kl_l"):
            self.kl_l = torch.zeros(self.K, device=self.device)
        if not hasattr(self, "kl_f"):
            self.kl_f = torch.zeros(self.K, device=self.device)

            # choose tau map depending on mode
        tau_map = None
        if self.type_noise == NoiseType.CONSTANT:
            # tau_scalar = float(self.tau.item())
            pass
        elif self.type_noise == NoiseType.ROW_WISE:
            tau_map = self.tau_map  # (N,P)
        elif self.type_noise == NoiseType.COLUMN_WISE:
            tau_map = self.tau_map  # (N,P)
        else:
            raise ValueError("Invalid type_noise")

        for k in range(self.K):
            self.update_factor(k, tau_map=tau_map, eps=eps)

        if self.allow_backfitting and self.K > 1:
            print(self.K)
            to_drop = [k for k in range(self.K) if self._should_prune_factor(k)]
            if len(to_drop) >= self.K:
                keep_one = min(to_drop)
                to_drop = [k for k in range(self.K) if k != keep_one]
            # drop highest indices first to avoid reindex churn
            to_drop_sorted = sorted(to_drop, reverse=True)
            self._prune_indices(to_drop_sorted)
        self.update_tau()
        self.cal_obj()

    def update_factor(self, k: int, tau_map: Tensor | None = None, eps: float = 1e-12) -> None:
        """
        Update L[:,k], F[:,k] and their second moments using the current priors.
        Handles scalar tau (tau_map=None) or elementwise tau (tau_map is (N,P)).
        """
        with torch.no_grad():  # ---------- Update L[:, k] ----------
            Rk = self._partial_residual_masked(k)  # (N,P), zeros where missing
            fk = self.F[:, k]  # (P,)
            fk2 = self.F2[:, k]  # (P,)

            if tau_map is None:
                denom_l = (fk2.view(1, -1) * self.mask).sum(dim=1).clamp_min(eps)  # (N,)
                num_l = Rk @ fk  # (N,)
                se_l = torch.sqrt(1.0 / (self.tau * denom_l))
            else:
                denom_l = (tau_map * (fk2.view(1, -1) * self.mask)).sum(dim=1).clamp_min(eps)  # (N,)
                num_l = (tau_map * Rk) @ fk  # (N,)
                se_l = torch.sqrt(1.0 / denom_l)

            lhat = num_l / denom_l
        # print(denom_l)

        X_model = self.update_cov_L(k)
        with torch.enable_grad():
            resL = self.prior_L_fn.fit(
                X=X_model,
                betahat=lhat,
                sebetahat=se_l,
                model_param=self.model_state_L[k],
            )
        with torch.no_grad():
            self.model_state_L[k] = resL.model_param
            self.L[:, k] = resL.post_mean
            self.L2[:, k] = resL.post_mean2
            nm_ll_L = normal_means_loglik(x=lhat, s=se_l, Et=resL.post_mean, Et2=resL.post_mean2)
            self.kl_l[k] = torch.as_tensor((-resL.loss) - nm_ll_L, device=self.device)
            self.pi0_L[k] = resL.pi0_null if hasattr(resL, "pi0_null") else None

            # ---------- Update F[:, k] ----------
            Rk = self._partial_residual_masked(k)  # recompute with updated L
            lk = self.L[:, k]  # (N,)
            lk2 = self.L2[:, k]  # (N,)

            if tau_map is None:
                denom_f = (lk2.view(-1, 1) * self.mask).sum(dim=0).clamp_min(eps)  # (P,)
                num_f = Rk.T @ lk  # (P,)
                se_f = torch.sqrt(1.0 / (self.tau * denom_f))
            else:
                denom_f = (tau_map * (lk2.view(-1, 1) * self.mask)).sum(dim=0).clamp_min(eps)  # (P,)
                num_f = (tau_map * Rk).T @ lk  # (P,)
                se_f = torch.sqrt(1.0 / denom_f)

            fhat = num_f / denom_f

        X_model = self.update_cov_F(k)
        with torch.enable_grad():
            resF = self.prior_F_fn.fit(
                X=X_model,
                betahat=fhat,
                sebetahat=se_f,
                model_param=self.model_state_F[k],
            )
        with torch.no_grad():
            self.model_state_F[k] = resF.model_param
            self.F[:, k] = resF.post_mean
            self.F2[:, k] = resF.post_mean2
            # store as scalar on device; PriorResult.loss already = -log_lik
            nm_ll_F = normal_means_loglik(x=fhat, s=se_f, Et=resF.post_mean, Et2=resF.post_mean2)
            self.kl_f[k] = torch.as_tensor((-resF.loss) - nm_ll_F, device=self.device)
            self.pi0_F[k] = resF.pi0_null if hasattr(resF, "pi0_null") else None

    def cal_obj(self):
        # Data term
        ER2 = self.expected_residuals_squared()
        if self.type_noise == NoiseType.CONSTANT:
            m = self.mask.sum().clamp_min(1.0)
            ll = -0.5 * (
                m * (torch.log(torch.tensor(2 * torch.pi, device=self.device)) - torch.log(self.tau))
                + self.tau * ER2.sum()
            )
        else:
            obs = self.mask.bool()
            ll = -0.5 * (
                torch.log(torch.tensor(2 * torch.pi, device=self.device)) * obs.sum()
                - torch.log(self.tau_map[obs]).sum()
                + (self.tau_map * ER2)[obs].sum()
            )
        KL = self.kl_l.sum() + self.kl_f.sum()
        loss = (-ll + KL).item()  # minimize this (negative ELBO)
        self.obj.append(loss)

    @torch.no_grad()
    def update_cov_L(self, k: int):
        if self.self_row_cov:
            if self.X_l is None:
                if self.K > 1:
                    # all columns except k
                    others = self.L[:, torch.arange(self.K, device=self.device) != k]
                    X_model = others
                else:
                    X_model = self.L.new_ones(self.N, 1)  # intercept
            else:
                if self.K > 1:
                    others = self.L[:, torch.arange(self.K, device=self.device) != k]
                    X_model = torch.hstack((self.X_l, others))
                else:
                    X_model = self.X_l
        else:
            X_model = self.X_l
        return X_model

    @torch.no_grad()
    def update_cov_F(self, k: int):
        if self.self_col_cov:
            if self.X_f is None:
                if self.K > 1:
                    others = self.F[:, torch.arange(self.K, device=self.device) != k]
                    X_model = others
                else:
                    X_model = self.F.new_ones(self.P, 1)
            else:
                if self.K > 1:
                    others = self.F[:, torch.arange(self.K, device=self.device) != k]
                    X_model = torch.hstack((self.X_f, others))
                else:
                    X_model = self.X_f
        else:
            X_model = self.X_f
        return X_model

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
        Yfit = self.L @ self.F.T  # (N,P)
        resid_mean_sq = (self.Y0 - Yfit).pow(2)  # (N,P)
        first_moment_sq = (self.L.pow(2)) @ (self.F.pow(2)).T  # Σ_k (E[L]^2)(E[F]^2)^T
        second_moment = self.L2 @ self.F2.T  # Σ_k E[L^2] E[F^2]^T
        R2 = resid_mean_sq - first_moment_sq + second_moment
        R2 = (R2 * self.mask).clamp_min(0.0)  # zero where missing; no negatives
        return R2

    def _pi0_min_value(self, pi0_val) -> float:
        if pi0_val is None:
            return float("-inf")
        if isinstance(pi0_val, torch.Tensor):
            if pi0_val.numel() == 0:
                return float("-inf")
            return float(pi0_val.min().item())
        return float(pi0_val)

    def _should_prune_factor(self, k: int) -> bool:
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
        return (pi0_min_L >= self.prune_thresh) or (pi0_min_F >= self.prune_thresh)

    def _prune_indices(self, idxs: list[int]) -> None:
        """In-place prune of K and all factor-aligned structures."""
        if not idxs:
            return
        keep = [i for i in range(self.K) if i not in idxs]
        self.L = self.L[:, keep]
        self.L2 = self.L2[:, keep]
        self.F = self.F[:, keep]
        self.F2 = self.F2[:, keep]
        self.kl_l = self.kl_l[keep]
        self.kl_f = self.kl_f[keep]
        self.model_state_L = [self.model_state_L[i] for i in keep]
        self.model_state_F = [self.model_state_F[i] for i in keep]
        self.pi0_L = [self.pi0_L[i] for i in keep]
        self.pi0_F = [self.pi0_F[i] for i in keep]
        self.K = len(keep)
        self.obj = []


def normal_means_loglik(
    x: torch.Tensor,
    s: torch.Tensor,
    Et: torch.Tensor,
    Et2: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduce: str = "sum",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Expected normal-means log-likelihood:
      E_q[ log N(x | theta, s^2) ] with q giving Et, Et2.

    Args:
      x, s, Et, Et2 : broadcastable tensors (same shape after broadcast).
      mask          : optional bool mask; True = include entry.
      reduce        : 'sum' (default), 'mean', or 'none' (per-element with NaNs for excluded).
      eps           : numerical floor for variance.

    Returns:
      Scalar if reduce in {'sum','mean'}, else elementwise tensor.
    """
    # Ensure common dtype/device via broadcasting
    x, s, Et, Et2 = torch.broadcast_tensors(x, s, Et, Et2)

    # Validity mask: finite & s > 0
    valid = torch.isfinite(x) & torch.isfinite(s) & torch.isfinite(Et) & torch.isfinite(Et2) & (s > 0)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool, device=x.device)

    if not valid.any():
        if reduce == "none":
            return torch.full_like(x, float("nan"))
        return x.new_tensor(float("nan"))

    # Stable variance and constant term
    var = (s * s).clamp_min(eps)  # s^2 ≥ eps
    c2pi = x.new_tensor(math.log(2.0 * math.pi))  # stays on same device/dtype

    # E[(x - theta)^2] = Et2 - 2*x*Et + x^2
    quad = Et2 - 2.0 * x * Et + x * x
    ll_el = -0.5 * (c2pi + torch.log(var) + quad / var)

    if reduce == "sum":
        return ll_el[valid].sum()
    elif reduce == "mean":
        return ll_el[valid].mean()
    elif reduce == "none":
        out = torch.full_like(x, float("nan"))
        out[valid] = ll_el[valid]
        return out
    else:
        raise ValueError("reduce must be 'sum', 'mean', or 'none'")
