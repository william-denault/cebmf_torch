import math
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Protocol

import torch
from torch import Tensor

from cebmf_torch.priors import PRIOR_REGISTRY
from cebmf_torch.utils.device import get_device
from cebmf_torch.utils.maths import safe_tensor_to_float

# Add at top of file after imports:
NUMERICAL_EPS = 1e-12
RANDOM_INIT_SCALE = 0.01
DEFAULT_PRUNE_THRESH = 1 - 1e-3


# Initialization strategies
class InitialisationStrategy(Protocol):
    """Protocol for factor initialization strategies."""

    def __call__(self, Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Initialize L and F matrices.

        Args:
            Y: Input data tensor (N, P)
            N: Number of rows
            P: Number of columns
            K: Number of factors
            device: Target device

        Returns:
            Tuple of (L, F) tensors
        """
        ...


@torch.no_grad()
def svd_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """SVD-based initialization strategy."""
    Y_for_init = _impute_nan(Y)
    U, S, Vh = torch.linalg.svd(Y_for_init, full_matrices=False)
    K_actual = min(K, S.shape[0])
    L = (U[:, :K_actual] * S[:K_actual]).contiguous()
    F = Vh[:K_actual, :].T.contiguous()

    # Pad with zeros if K > rank
    if K_actual < K:
        L = torch.cat([L, torch.zeros(N, K - K_actual, device=device)], dim=1)
        F = torch.cat([F, torch.zeros(P, K - K_actual, device=device)], dim=1)

    return L, F


@torch.no_grad()
def random_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Random initialization strategy."""
    L = torch.randn(N, K, device=device) * RANDOM_INIT_SCALE
    F = torch.randn(P, K, device=device) * RANDOM_INIT_SCALE
    return L, F


@torch.no_grad()
def zero_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Zero initialization strategy."""
    L = torch.zeros(N, K, device=device)
    F = torch.zeros(P, K, device=device)
    return L, F


# Registry for initialization strategies
INIT_STRATEGIES = {
    "svd": svd_initialise,
    "random": random_initialise,
    "zero": zero_initialise,
}


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


@dataclass
class ModelParams:
    K: int = 5
    prior_L: str = "norm"
    prior_F: str = "norm"
    allow_backfitting: bool = True
    prune_thresh: float = DEFAULT_PRUNE_THRESH


@dataclass
class NoiseParams:
    type: NoiseType = NoiseType.CONSTANT


@dataclass
class CovariateParams:
    X_l: Tensor | None = None
    X_f: Tensor | None = None
    self_row_cov: bool = False
    self_col_cov: bool = False


@dataclass
class cEBMF:
    """
    Pure-PyTorch EBMF with proper NaN handling:
    - Observed-mask weighting in lhat/fhat and their standard errors.
    - Constant noise precision (scalar tau).
    - Mini-batch optimization for mixture weights inside ash().
    """

    data: Tensor
    model: ModelParams = field(default_factory=ModelParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    covariate: CovariateParams = field(default_factory=CovariateParams)
    device: torch.device = field(default_factory=get_device)

    def __post_init__(self):
        self._validate_inputs()
        self.device = self.device or get_device()
        self.Y = self.data.to(self.device).float()
        self.N, self.P = self.Y.shape
        self._initialise_priors()
        self._initialise_tensors()

    @torch.no_grad()
    def fit(self, maxit: int = 50):
        self.initialise_factors()
        for _ in range(maxit):
            self.iter_once()
        return CEBMFResult(self.L, self.F, self.tau, self.obj)

    @torch.no_grad()
    def initialise_factors(self, method: str = "svd"):
        if method not in INIT_STRATEGIES:
            raise ValueError(f"Unknown initialization method '{method}'. Available: {list(INIT_STRATEGIES.keys())}")

        initialise_fn = INIT_STRATEGIES[method]
        self.L, self.F = initialise_fn(self.Y, self.N, self.P, self.model.K, self.device)

        self.L2 = self.L * self.L
        self.F2 = self.F * self.F
        self.R = (self.Y0 - self.L @ self.F.T)
        self.R.mul_(self.mask)
        
        self.R.nan_to_num_(nan=0.0)
        self.update_tau()

    @torch.no_grad()
    def iter_once(self):
        tau_map = None if self.noise.type == NoiseType.CONSTANT else self.tau_map
        for k in range(self.model.K):
            self._update_factors(k, tau_map=tau_map, eps=NUMERICAL_EPS)

        self._backfit()
        self.update_tau()
        self._cal_obj()

    @torch.no_grad()
    def update_tau(self):
        """
        Matches NumPy behavior:
        - 'constant'   -> scalar tau; also provides tau_map (N,P) if you need it
        - 'row_wise'   -> tau_row (N,), tau_map broadcast to (N,P)
        - 'column_wise'-> tau_col (P,), tau_map broadcast to (N,P)
        """
        R2 = self._expected_residuals_squared()  # (N,P), zeros at missing

        match self.noise.type:
            case NoiseType.CONSTANT:
                dim = None
            case NoiseType.COLUMN_WISE:
                dim = 0
            case NoiseType.ROW_WISE:
                dim = 1
            case _:
                raise ValueError("type_noise must be 'constant', 'row_wise', or 'column_wise'")

        self._update_tau(R2, dim=dim)

    # =========================================================================
    # Private Methods - Internal Implementation Details
    # =========================================================================

    @torch.no_grad() 
    def _update_factors(self, k: int, tau_map: Tensor | None = None, eps: float = NUMERICAL_EPS) -> None:
        """Orchestrates residualization. Only this method mutates self.R."""
        mask_f = self.mask if self.mask.dtype.is_floating_point else self.mask.to(self.L.dtype)
 
        Lk = self.L[:, k]
        Fk = self.F[:, k]
 
        self.R.addr_(Lk, Fk, alpha=1.0)
        self.R.mul_(mask_f)
 
        self._update_L_factor(k, tau_map, eps)
 
        Lk = self.L[:, k]  # updated
        self.R.addr_(Lk, Fk, alpha=-1.0)
        self.R.mul_(mask_f)
 
        self.R.addr_(Lk, Fk, alpha=1.0)
        self.R.mul_(mask_f)
 
        self._update_F_factor(k, tau_map, eps)
 
        Fk = self.F[:, k]  # updated
        self.R.addr_(Lk, Fk, alpha=-1.0)
        self.R.mul_(mask_f)

    @torch.no_grad()
    def _update_L_factor(self, k: int, tau_map: Tensor | None, eps: float) -> None:
        """Update L[:,k] and L2[:,k]; assumes self.R already has k added back."""
        mask_f = self.mask if self.mask.dtype.is_floating_point else self.mask.to(self.L.dtype)
        Fk  = self.F[:, k]
        Fk2 = self.F2[:, k]

        if tau_map is None:
            denom_l = mask_f @ Fk2                   # (N,)
            num_l   = self.R @ Fk                    # (N,)
            se_l    = torch.sqrt(1.0 / (self.tau * denom_l.clamp_min(eps)))
        else:
            denom_l = (tau_map * mask_f) @ Fk2       # (N,)
            num_l   = torch.einsum('ij,ij,j->i', self.R, tau_map, Fk)
            se_l    = torch.sqrt(1.0 / denom_l.clamp_min(eps))

        lhat = num_l / denom_l.clamp_min(eps)

        # fit prior for L
        X_model = self._build_covariate_matrix(
            external_cov=self.covariate.X_l,
            self_cov_enabled=self.covariate.self_row_cov,
            factors=self.L,
            k=k,
            dim_size=self.N,
        )
        with torch.enable_grad():
            resL = self.prior_L_fn.fit(
                X=X_model, betahat=lhat, sebetahat=se_l, model_param=self.model_state_L[k],
            )

        # write back
        self.model_state_L[k] = resL.model_param
        self.L[:, k]  = resL.post_mean
        self.L2[:, k] = resL.post_mean2
        nm_ll_L = normal_means_loglik(x=lhat, s=se_l, Et=resL.post_mean, Et2=resL.post_mean2)
        self.kl_l[k]  = torch.as_tensor((-resL.loss) - nm_ll_L, device=self.device)
        self.pi0_L[k] = resL.pi0_null


    @torch.no_grad()
    def _update_F_factor(self, k: int, tau_map: Tensor | None, eps: float) -> None:
        """Update F[:,k] and F2[:,k]; assumes self.R has UPDATED L_k added back."""
        mask_f = self.mask if self.mask.dtype.is_floating_point else self.mask.to(self.L.dtype)
        Lk  = self.L[:, k]
        Lk2 = self.L2[:, k]

        if tau_map is None:
            denom_f = mask_f.T @ Lk2                 # (P,)
            num_f   = self.R.T @ Lk                  # (P,)
            se_f    = torch.sqrt(1.0 / (self.tau * denom_f.clamp_min(eps)))
        else:
            denom_f = (tau_map * mask_f).transpose(0, 1) @ Lk2
            num_f   = torch.einsum('ij,ij,i->j', self.R, tau_map, Lk)
            se_f    = torch.sqrt(1.0 / denom_f.clamp_min(eps))

        fhat = num_f / denom_f.clamp_min(eps)

    # fit prior for F
        X_model = self._build_covariate_matrix(
            external_cov=self.covariate.X_f,
            self_cov_enabled=self.covariate.self_col_cov,
            factors=self.F,
            k=k,
            dim_size=self.P,
        )
        with torch.enable_grad():
            resF = self.prior_F_fn.fit(
                X=X_model, betahat=fhat, sebetahat=se_f, model_param=self.model_state_F[k],
            )

        # write back
        self.model_state_F[k] = resF.model_param
        self.F[:, k]  = resF.post_mean
        self.F2[:, k] = resF.post_mean2
        nm_ll_F = normal_means_loglik(x=fhat, s=se_f, Et=resF.post_mean, Et2=resF.post_mean2)
        self.kl_f[k]  = torch.as_tensor((-resF.loss) - nm_ll_F, device=self.device)
        self.pi0_F[k] = resF.pi0_null

    @torch.no_grad()
    def _cal_obj(self):
        # Data term
        ER2 = self._expected_residuals_squared()
        if self.noise.type == NoiseType.CONSTANT:
            ll = self._compute_constant_loglik(ER2)
        else:
            ll = self._compute_elementwise_loglik(ER2)

        KL = self.kl_l.sum() + self.kl_f.sum()
        loss = (-ll + KL).item()  # minimize this (negative ELBO)
        self.obj.append(loss)

    @torch.no_grad()
    def _compute_constant_loglik(self, ER2: Tensor) -> Tensor:
        m = self.mask.sum().clamp_min(1.0)
        return -0.5 * (
            m * (torch.log(torch.tensor(2 * torch.pi, device=self.device)) - torch.log(self.tau))
            + self.tau * ER2.sum()
        )

    @torch.no_grad()
    def _compute_elementwise_loglik(self, ER2: Tensor) -> Tensor:
        obs = self.mask.bool()
        return -0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=self.device)) * obs.sum()
            - torch.log(self.tau_map[obs]).sum()
            + (self.tau_map * ER2)[obs].sum()
        )

    @torch.no_grad()
    def _backfit(self):
        if not (self.model.allow_backfitting and self.model.K > 1):
            return

        to_drop = [k for k in range(self.model.K) if self._should_prune_factor(k)]
        if len(to_drop) >= self.model.K:
            keep_one = min(to_drop)
            to_drop = [k for k in range(self.model.K) if k != keep_one]
        # drop highest indices first to avoid reindex churn
        to_drop_sorted = sorted(to_drop, reverse=True)
        self._prune_indices(to_drop_sorted)

    @torch.no_grad()
    def _update_fitted_value(self):
        self.Y_fit = self.L @ self.F.T

    @torch.no_grad()
    def _expected_residuals_squared(self):
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

    @torch.no_grad()
    def _validate_inputs(self) -> None:
        if self.model.K < 1:
            raise ValueError(f"K must be >= 1, got {self.model.K}")
        if torch.isnan(self.data).all():
            raise ValueError("Data cannot be all NaN")
        # More validation...

    @torch.no_grad()
    def _initialise_priors(self):
        self.prior_L_fn = PRIOR_REGISTRY.get_builder(self.model.prior_L)
        self.prior_F_fn = PRIOR_REGISTRY.get_builder(self.model.prior_F)
        self.model_state_L = [None] * self.model.K
        self.model_state_F = [None] * self.model.K

    @torch.no_grad()
    def _initialise_tensors(self):
        self.mask = (~torch.isnan(self.Y)).float()  # 1 where observed, 0 where NaN
        self.Y0 = torch.nan_to_num(self.Y, nan=0.0)  # zeros where missing
        self.L = torch.zeros(self.N, self.model.K, device=self.device)
        self.L2 = torch.zeros(self.N, self.model.K, device=self.device)
        self.F = torch.zeros(self.P, self.model.K, device=self.device)
        self.F2 = torch.zeros(self.P, self.model.K, device=self.device)
        self.tau = torch.tensor(1.0, device=self.device)  # precision (1/var)
        self.kl_l = torch.zeros(self.model.K, device=self.device)
        self.kl_f = torch.zeros(self.model.K, device=self.device)
        self.pi0_L: list[Tensor | float | None] = [
            None
        ] * self.model.K  # store latest pi0 for L[:,k]; scalar or Tensor or None
        self.pi0_F: list[Tensor | float | None] = [None] * self.model.K
        self.obj = []

    @torch.no_grad()
    def _update_tau(self, R2: Tensor, dim: None | int) -> None:
        if dim not in (None, 0, 1):
            raise ValueError("dim must be None, 0, or 1")

        m = self.mask.sum(dim=dim).clamp_min(1.0)
        mean_R2 = R2.sum(dim=dim) / m
        tau = 1.0 / (mean_R2)

        if dim is None:
            self.tau = tau  # scalar (back-compat)
            self.tau_map = torch.full((self.N, self.P), tau.item(), device=self.device, dtype=R2.dtype)
            return

        view = (-1, 1) if dim == 1 else (1, -1)
        self.tau_map = tau.view(*view).expand(self.N, self.P)  # (N,P)
        self.tau = self.tau_map  # if downstream expects elementwise

    @torch.no_grad()
    def _partial_residual_masked(self, k: int) -> Tensor:
        # Rk for observed entries only
        recon = self.L @ self.F.T
        k_contrib = torch.outer(self.L[:, k], self.F[:, k])
        Rk = (self.Y0 - (recon - k_contrib)) * self.mask
        return Rk

    @torch.no_grad()
    def _should_prune_factor(self, k: int) -> bool:
        """
        Remove factor k if we have π₀ info and the smallest π₀ across coordinates
        (for any side that provided it) is ≥ thresh. (Your spec: use the *lowest* π₀.)
        """
        pi0_min_L = safe_tensor_to_float(self.pi0_L[k])
        pi0_min_F = safe_tensor_to_float(self.pi0_F[k])
        # if neither side provided π0, don't prune
        if pi0_min_L == float("-inf") and pi0_min_F == float("-inf"):
            return False
        # If either side indicates "all near spike", prune.
        thresh = self.model.prune_thresh
        return (pi0_min_L >= thresh) or (pi0_min_F >= thresh)

    @torch.no_grad()
    def _prune_indices(self, idxs: list[int]) -> None:
        """In-place prune of K and all factor-aligned structures."""
        if not idxs:
            return
        keep = [i for i in range(self.model.K) if i not in idxs]
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
        self.model.K = len(keep)
        self.obj = []

    @torch.no_grad()
    def _build_covariate_matrix(
        self, external_cov: Tensor | None, self_cov_enabled: bool, factors: Tensor, k: int, dim_size: int
    ) -> Tensor | None:
        """Build covariate matrix combining external and self-covariates."""
        if not self_cov_enabled:
            return external_cov

        # Get other factors (excluding k)
        if self.model.K > 1:
            others = factors[:, torch.arange(self.model.K, device=self.device) != k]
            if external_cov is None:
                return others
            return torch.hstack((external_cov, others))

        # K=1 case: return external covariates or intercept
        return external_cov if external_cov is not None else factors.new_ones(dim_size, 1)


def normal_means_loglik(
    x: torch.Tensor,
    s: torch.Tensor,
    Et: torch.Tensor,
    Et2: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduce: str = "sum",
    eps: float = NUMERICAL_EPS,
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


@torch.no_grad()
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
