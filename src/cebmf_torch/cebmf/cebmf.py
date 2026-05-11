import math
from dataclasses import dataclass
from enum import StrEnum, auto
from warnings import warn

import torch
from torch import Tensor

from cebmf_torch.cebmf._initialisation import INIT_STRATEGIES, user_provided_factors
from cebmf_torch.priors import PRIOR_REGISTRY
from cebmf_torch.utils.device import get_device
from cebmf_torch.utils.maths import safe_tensor_to_float

# Add at top of file after imports:
NUMERICAL_EPS = 1e-12
DEFAULT_PRUNE_THRESH = 1 - 1e-3


@dataclass
class CEBMFResult:
    """
    Container for cEBMF results.

    Attributes
    ----------
    L : Tensor
        Left factor matrix.
    F : Tensor
        Right factor matrix.
    tau : Tensor
        Noise precision(s).
    history_obj : list
        History of objective values.
    """

    L: Tensor
    F: Tensor
    tau: Tensor
    history_obj: list


class NoiseType(StrEnum):
    CONSTANT = auto()
    ROW_WISE = auto()
    COLUMN_WISE = auto()
    KNOWN = auto()  # user-supplied standard errors; variance is fixed (not learned)


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


class cEBMF:
    """
    Pure-PyTorch Empirical Bayes Matrix Factorization (EBMF) with NaN handling.

    Features
    --------
    - Observed-mask weighting in lhat/fhat and their standard errors.
    - Constant or structured noise precision (scalar or per-row/column tau).
    - User-supplied (fixed) standard errors via ``S`` — useful for z-scores
      (pass ``S=1.0``) or pre-computed standard errors of effect-size estimates
      (pass an ``(N, P)`` tensor). When ``S`` is provided, the noise variance is
      treated as known and is *not* re-estimated during fitting.
    - Mini-batch optimization for mixture weights inside ash().
    - Modular prior and covariate support.
    """

    def __init__(
        self,
        data: torch.Tensor,
        K: int = 5,
        prior_L: str = "norm",
        prior_F: str = "norm",
        internal_epoch: int = 10,
        prior_L_kwargs: dict | None = None,
        prior_F_kwargs: dict | None = None,
        allow_backfitting: bool = True,
        prune_thresh: float = DEFAULT_PRUNE_THRESH,
        noise_type: NoiseType = NoiseType.CONSTANT,
        S: torch.Tensor | float | int | None = None,
        X_l: torch.Tensor | None = None,
        X_f: torch.Tensor | None = None,
        self_row_cov: bool = False,
        self_col_cov: bool = False,
        device: torch.device | None = None,
    ):
        """
        Parameters
        ----------
        data : torch.Tensor
            Observed data matrix of shape (N, P). NaN entries are treated as missing.
        K : int, optional
            Initial number of factors. Default 5.
        prior_L, prior_F : str, optional
            Prior names to use for the row/column factors.
        internal_epoch : int, optional
            Number of inner epochs for the prior fitting routine.
        prior_L_kwargs, prior_F_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the prior builders.
        allow_backfitting : bool, optional
            If True, allow factor pruning between iterations.
        prune_thresh : float, optional
            π0 threshold above which a factor is pruned.
        noise_type : NoiseType, optional
            Structure of the (learned) noise variance. Ignored if ``S`` is provided.
        S : torch.Tensor, float, int, or None, optional
            User-supplied standard errors. When provided, the noise variance is
            taken as fixed (not estimated) and ``noise_type`` is forced to
            :attr:`NoiseType.KNOWN`. ``S`` may be:

            * a scalar — typical for z-scores (``S=1.0``) — applied to every
              entry;
            * a tensor broadcastable to ``data.shape`` — for example, ``(N, P)``
              if every observation has its own standard error, or ``(P,)`` /
              ``(N, 1)`` for column- or row-wise known SEs.

            Entries of ``S`` that are NaN, infinite, or non-positive are folded
            into the missing-data mask (those observations are dropped from the
            likelihood). A warning is raised if any such entries align with
            observed values in ``data``.
        X_l, X_f : torch.Tensor or None, optional
            External covariates for the row/column factors.
        self_row_cov, self_col_cov : bool, optional
            Whether to use other factors as self-covariates.
        device : torch.device or None, optional
            Target device. Defaults to the result of :func:`get_device`.
        """
        self.data = data
        self.device = device or get_device()

        # If the user supplied S, the noise variance is fixed and any explicit
        # ``noise_type`` other than the default CONSTANT is silently overridden.
        if S is not None:
            if noise_type != NoiseType.CONSTANT:
                warn(
                    f"`S` was provided so the noise variance is treated as known/fixed; "
                    f"the requested noise_type={noise_type!r} will be ignored.",
                    stacklevel=2,
                )
            noise_type = NoiseType.KNOWN

        # Build config objects internally
        self.model = ModelParams(
            K=K, prior_L=prior_L, prior_F=prior_F, allow_backfitting=allow_backfitting, prune_thresh=prune_thresh
        )
        self.noise = NoiseParams(type=noise_type)
        # Move covariates to device (if provided) to avoid later CPU↔GPU hops
        self.covariate = CovariateParams(
            X_l=(X_l.to(self.device) if X_l is not None else None),
            X_f=(X_f.to(self.device) if X_f is not None else None),
            self_row_cov=self_row_cov,
            self_col_cov=self_col_cov,
        )

        if prior_L_kwargs is None:
            prior_L_kwargs = {}
        if prior_F_kwargs is None:
            prior_F_kwargs = {}

        # Stash raw S input; normalised to an (N, P) tensor inside _initialise_tensors
        self._S_input = S
        self._validate_inputs()
        self.Y = self.data.to(self.device).float()
        self.N, self.P = self.Y.shape
        self._initialise_priors(prior_L_kwargs=prior_L_kwargs, prior_F_kwargs=prior_F_kwargs)
        self._initialise_tensors()
        self.internal_epoch = internal_epoch
        self._factors_initialised = False

    @torch.no_grad()
    def fit(self, maxit: int = 50):
        """
        Fit the cEBMF model for a specified number of iterations.

        Parameters
        ----------
        maxit : int, optional
            Number of iterations to run. Default is 50.

        Returns
        -------
        CEBMFResult
            Result container with fitted factors, noise, and objective history.
        """
        if not self._factors_initialised:
            warn("Factors not initialized; using SVD initialization.", stacklevel=2)
            self.initialise_factors()
        for _ in range(maxit):
            self.iter_once()
        return CEBMFResult(self.L, self.F, self.tau, self.obj)

    @torch.no_grad()
    def initialise_factors(self, method: str = "svd", *, L: Tensor | None = None, F: Tensor | None = None):
        """
        Initialize factor matrices using the specified method, or user-provided initial factors.

        Parameters
        ----------
        method : str
            Initialization method ('svd', 'random', or 'zero'). Default is 'svd'. Ignored if L and F are provided.
        L : Tensor or None, optional
            User-provided initial factor matrix (N, K).  Ignored if F not also provided.
        F : Tensor or None, optional
            User-provided initial factor matrix (P, K).  Ignored if L not also provided.
        """

        def _use_strategy(method: str):
            initialise_fn = INIT_STRATEGIES[method]
            self.L, self.F = initialise_fn(self.Y, self.N, self.P, self.model.K, self.device)

        if L is None and F is not None:
            warn("Provided F without L; ignoring F and using svd for initialization.", stacklevel=2)
            _use_strategy("svd")
        elif L is not None and F is None:
            warn("Provided L without F; ignoring L and using svd for initialization.", stacklevel=2)
            _use_strategy("svd")
        elif L is not None and F is not None:
            self.L, self.F = user_provided_factors(L, F, self.N, self.P, self.model.K, self.device)

        elif method not in INIT_STRATEGIES:
            raise ValueError(f"Unknown initialization method '{method}'. Available: {list(INIT_STRATEGIES.keys())}")
        else:
            _use_strategy(method)

        self.L2 = self.L * self.L
        self.F2 = self.F * self.F
        self.R = self.Y0 - self.L @ self.F.T
        self.R.mul_(self.mask)

        self.R.nan_to_num_(nan=0.0)
        self.update_tau()

        self._factors_initialised = True

    @torch.no_grad()
    def iter_once(self):
        """
        Perform one iteration of the cEBMF update (update all factors and noise).
        """
        tau_map = None if self.noise.type == NoiseType.CONSTANT else self.tau_map
        for k in range(self.model.K):
            self._update_factors(k, tau_map=tau_map, eps=NUMERICAL_EPS)

        self.update_tau()
        self._backfit()

        self._cal_obj()

    @torch.no_grad()
    def update_tau(self):
        """
        Update the noise precision parameter(s) according to the noise model.

        Behaviour by noise type:

        - ``CONSTANT``    -> scalar tau; also provides tau_map (N,P) if you need it
        - ``ROW_WISE``    -> tau_row (N,), tau_map broadcast to (N,P)
        - ``COLUMN_WISE`` -> tau_col (P,), tau_map broadcast to (N,P)
        - ``KNOWN``       -> no-op; tau_map was built once from the user-supplied ``S``.
        """
        if self.noise.type == NoiseType.KNOWN:
            # Variance is fixed by the user; nothing to update.
            return

        R2 = self._expected_residuals_squared()  # (N,P), zeros at missing

        match self.noise.type:
            case NoiseType.CONSTANT:
                dim = None
            case NoiseType.COLUMN_WISE:
                dim = 0
            case NoiseType.ROW_WISE:
                dim = 1
            case _:
                raise ValueError("type_noise must be 'constant', 'row_wise', 'column_wise', or 'known'")

        self._update_tau(R2, dim=dim)

    # =========================================================================
    # Private Methods - Internal Implementation Details
    # =========================================================================

    @torch.no_grad()
    def _update_factors(self, k: int, tau_map: Tensor | None = None, eps: float = NUMERICAL_EPS) -> None:
        """Update L[:,k] then F[:,k] using stable, non-inplace residualization."""
        # Full residual with current factors (exclude all k implicitly)
        self._recompute_residual()

        # --- L update uses Rk = R + L_k F_k^T (add back k's current contrib)
        Rk = self.R + torch.outer(self.L[:, k], self.F[:, k])
        Rk.mul_(self.mask)
        self._update_L_factor(k, Rk, tau_map, eps)

        # After updating L, refresh residual with new L_k
        self._recompute_residual()

        # --- F update with updated L: again build Rk
        Rk = self.R + torch.outer(self.L[:, k], self.F[:, k])
        Rk.mul_(self.mask)
        self._update_F_factor(k, Rk, tau_map, eps)

        # Final residual for next factor
        self._recompute_residual()

    @torch.no_grad()
    def _update_L_factor(self, k: int, Rk: Tensor, tau_map: Tensor | None, eps: float) -> None:
        """Update L[:,k] and L2[:,k] using provided Rk (residual with k added back)."""
        mask_f = self.mask if self.mask.dtype.is_floating_point else self.mask.to(self.L.dtype)
        Fk = self.F[:, k]
        Fk2 = self.F2[:, k]

        if tau_map is None:
            denom_l = (mask_f @ Fk2).clamp_min(eps)  # (N,)
            num_l = Rk @ Fk  # (N,)
            se_l = torch.sqrt(1.0 / (self.tau * denom_l))
        else:
            W = tau_map * mask_f
            denom_l = (W @ Fk2).clamp_min(eps)  # (N,)
            num_l = (W * Rk) @ Fk  # (N,)
            se_l = torch.sqrt(1.0 / denom_l)

        lhat = num_l / denom_l

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
                X=X_model,
                betahat=lhat,
                internal_epoch=self.internal_epoch,
                sebetahat=se_l,
                model_param=self.model_state_L[k],
                device=self.device,
            )

        # write back
        self.model_state_L[k] = resL.model_param
        self.L[:, k] = resL.post_mean
        self.L2[:, k] = resL.post_mean2
        nm_ll_L = normal_means_loglik(x=lhat, s=se_l, Et=resL.post_mean, Et2=resL.post_mean2)
        self.kl_l[k] = torch.as_tensor((-resL.loss) - nm_ll_L, device=self.device, dtype=self.L.dtype)
        self.pi0_L[k] = resL.pi0_null

    @torch.no_grad()
    def _update_F_factor(self, k: int, Rk: Tensor, tau_map: Tensor | None, eps: float) -> None:
        """Update F[:,k] and F2[:,k] using provided Rk (residual with k added back)."""
        mask_f = self.mask if self.mask.dtype.is_floating_point else self.mask.to(self.L.dtype)
        Lk = self.L[:, k]
        Lk2 = self.L2[:, k]

        if tau_map is None:
            denom_f = (mask_f.T @ Lk2).clamp_min(eps)  # (P,)
            num_f = Rk.T @ Lk  # (P,)
            se_f = torch.sqrt(1.0 / (self.tau * denom_f))
        else:
            W = tau_map * mask_f
            denom_f = (W.T @ Lk2).clamp_min(eps)
            num_f = (W * Rk).transpose(0, 1) @ Lk
            se_f = torch.sqrt(1.0 / denom_f)

        fhat = num_f / denom_f

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
                X=X_model,
                betahat=fhat,
                sebetahat=se_f,
                internal_epoch=self.internal_epoch,
                model_param=self.model_state_F[k],
                device=self.device,
            )

        # write back
        self.model_state_F[k] = resF.model_param
        self.F[:, k] = resF.post_mean
        self.F2[:, k] = resF.post_mean2
        nm_ll_F = normal_means_loglik(x=fhat, s=se_f, Et=resF.post_mean, Et2=resF.post_mean2)
        self.kl_f[k] = torch.as_tensor((-resF.loss) - nm_ll_F, device=self.device, dtype=self.F.dtype)
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
        c2pi = ER2.new_tensor(math.log(2.0 * math.pi))
        # tau is scalar-precision in this branch
        return -0.5 * (m * (c2pi - torch.log(self.tau)) + self.tau * ER2.sum())

    @torch.no_grad()
    def _compute_elementwise_loglik(self, ER2: Tensor) -> Tensor:
        obs = self.mask.bool()
        c2pi = ER2.new_tensor(math.log(2.0 * math.pi))
        return -0.5 * (c2pi * obs.sum() - torch.log(self.tau_map[obs]).sum() + (self.tau_map * ER2)[obs].sum())

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
        # R2 = resid_mean_sq - first_moment_sq + second_moment
        # R2 = (R2 * self.mask).clamp_min(0.0)  # zero where missing; no negatives

        variance_term = second_moment - first_moment_sq

        R2 = resid_mean_sq + variance_term  # NOT minus!
        R2 = (R2 * self.mask).clamp_min(0.0)
        return R2

    @torch.no_grad()
    def _validate_inputs(self) -> None:
        if self.model.K < 1:
            raise ValueError(f"K must be >= 1, got {self.model.K}")
        if torch.isnan(self.data).all():
            raise ValueError("Data cannot be all NaN")
        # More validation...

    @torch.no_grad()
    def _initialise_priors(self, prior_L_kwargs: dict, prior_F_kwargs: dict) -> None:
        self.prior_L_fn = PRIOR_REGISTRY.get_builder(self.model.prior_L)
        self.prior_L_fn.set_kwargs(**prior_L_kwargs)
        self.prior_F_fn = PRIOR_REGISTRY.get_builder(self.model.prior_F)
        self.prior_F_fn.set_kwargs(**prior_F_kwargs)
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

        if self.noise.type == NoiseType.KNOWN:
            # Build tau_map from user-supplied S; may further reduce self.mask / self.Y0.
            self._setup_known_variance()
        else:
            # Initial precision guess for learned-noise modes; refined by update_tau().
            self.S = None
            self.tau = torch.tensor(1.0, device=self.device)

        self.kl_l = torch.zeros(self.model.K, device=self.device)
        self.kl_f = torch.zeros(self.model.K, device=self.device)
        self.pi0_L: list[Tensor | float | None] = [
            None
        ] * self.model.K  # store latest pi0 for L[:,k]; scalar or Tensor or None
        self.pi0_F: list[Tensor | float | None] = [None] * self.model.K
        self.obj = []

    @torch.no_grad()
    def _setup_known_variance(self) -> None:
        """
        Materialise the user-supplied standard-error matrix into ``self.S`` and
        the corresponding precision matrix ``self.tau_map = 1 / S**2``.

        Accepts a scalar (broadcast to ``(N, P)``) or any tensor broadcastable to
        ``(N, P)``. Entries of S that are NaN, infinite, or non-positive are folded
        into ``self.mask`` (treated as missing) and replaced with 1 internally so
        downstream multiplications by ``mask=0`` cleanly zero out their contribution.
        """
        S_in = self._S_input
        target_shape = (self.N, self.P)
        target_dtype = self.Y.dtype

        # --- Normalise S to an (N, P) tensor on the right device/dtype.
        if isinstance(S_in, (int, float)):
            S_val = float(S_in)
            if not math.isfinite(S_val) or S_val <= 0.0:
                raise ValueError(f"Scalar S must be a positive, finite number, got {S_in!r}")
            S = torch.full(target_shape, S_val, device=self.device, dtype=target_dtype)
        else:
            if not isinstance(S_in, torch.Tensor):
                S = torch.as_tensor(S_in, dtype=target_dtype, device=self.device)
            else:
                S = S_in.to(device=self.device, dtype=target_dtype)
            if S.shape != target_shape:
                try:
                    S = S.expand(target_shape).contiguous()
                except RuntimeError as e:
                    raise ValueError(
                        f"S has shape {tuple(S.shape)}, which is not broadcastable to data shape {target_shape}"
                    ) from e
            else:
                S = S.contiguous()

        # --- Fold non-finite / non-positive entries into the mask.
        bad = ~torch.isfinite(S) | (S <= 0)
        if bool(bad.any()):
            # Only warn about bad-S entries that align with currently-observed data:
            # those are the ones that actually change the likelihood.
            observed_and_bad = bad & self.mask.bool()
            n_observed_bad = int(observed_and_bad.sum().item())
            if n_observed_bad > 0:
                warn(
                    f"S has {n_observed_bad} non-finite or non-positive entries "
                    f"at observed positions in `data`; these entries will be treated "
                    f"as missing.",
                    stacklevel=3,
                )
            # Drop them from the mask and zero the corresponding Y0 entries.
            good_f = (~bad).to(self.mask.dtype)
            self.mask = self.mask * good_f
            self.Y0 = self.Y0 * self.mask
            # Replace bad S entries with 1.0 internally (any finite positive value works
            # because mask=0 there annihilates their contribution).
            S = torch.where(bad, torch.ones_like(S), S)

        self.S = S
        self.tau_map = 1.0 / (S * S)
        # Downstream code (e.g. _update_factors elementwise branch, _cal_obj
        # elementwise branch) reads self.tau_map; set self.tau to the same tensor
        # for consistency with the structured-noise convention.
        self.tau = self.tau_map

    @torch.no_grad()
    def _update_tau(self, R2: Tensor, dim: None | int) -> None:
        if dim not in (None, 0, 1):
            raise ValueError("dim must be None, 0, or 1")

        m = self.mask.sum(dim=dim).clamp_min(1.0)
        mean_R2 = R2.sum(dim=dim) / m
        tau = 1.0 / (mean_R2.clamp_min(NUMERICAL_EPS))

        if dim is None:
            # Scalar precision. Keep `self.tau` as a 0-d tensor (the loglik branch
            # for CONSTANT noise uses it directly) and expose a broadcast-only
            # tau_map for convenience. `expand` shares storage and avoids both
            # the per-iteration host sync from `tau.item()` and the (N, P)
            # materialisation that the previous `torch.full` triggered.
            self.tau = tau  # 0-d tensor
            self.tau_map = tau.view(1, 1).expand(self.N, self.P)  # (N, P) view, no copy
            return

        view = (-1, 1) if dim == 1 else (1, -1)
        self.tau_map = tau.view(*view).expand(self.N, self.P)  # (N,P)
        self.tau = self.tau_map  # downstream uses elementwise in structured noise

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
        if external_cov is not None and external_cov.device != self.device:
            external_cov = external_cov.to(self.device)

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

    @torch.no_grad()
    def _recompute_residual(self) -> None:
        """R := (Y0 - L F^T) ⊙ mask, NaNs -> 0."""
        self.R = self.Y0 - self.L @ self.F.T
        self.R.mul_(self.mask)
        self.R.nan_to_num_(nan=0.0)


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
    Compute the expected normal-means log-likelihood.

    E_q[ log N(x | theta, s^2) ] with q giving Et, Et2.

    Parameters
    ----------
    x : torch.Tensor
        Observed data (broadcastable with s, Et, Et2).
    s : torch.Tensor
        Standard errors (broadcastable).
    Et : torch.Tensor
        Posterior means (broadcastable).
    Et2 : torch.Tensor
        Posterior second moments (broadcastable).
    mask : torch.Tensor or None, optional
        Optional bool mask; True = include entry.
    reduce : str, optional
        'sum' (default), 'mean', or 'none' (per-element with NaNs for excluded).
    eps : float, optional
        Numerical floor for variance.

    Returns
    -------
    torch.Tensor
        Scalar if reduce in {'sum','mean'}, else elementwise tensor.
    """
    # Ensure common dtype/device via broadcasting
    x, s, Et, Et2 = torch.broadcast_tensors(x, s, Et, Et2)

    # Validity mask: finite & s > 0
    valid = torch.isfinite(x) & torch.isfinite(s) & torch.isfinite(Et) & torch.isfinite(Et2) & (s > 0)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool, device=x.device)

    # Stable variance and constant term. Use a safe `s` so log/quad never see 0
    # at masked entries; the masking happens after via `torch.where`.
    safe_s = torch.where(valid, s, torch.ones_like(s))
    var = (safe_s * safe_s).clamp_min(eps)
    c2pi = x.new_tensor(math.log(2.0 * math.pi))  # stays on same device/dtype

    # E[(x - theta)^2] = Et2 - 2*x*Et + x^2 (uses safe_x to keep finite)
    safe_x = torch.where(valid, x, torch.zeros_like(x))
    safe_Et = torch.where(valid, Et, torch.zeros_like(Et))
    safe_Et2 = torch.where(valid, Et2, torch.zeros_like(Et2))
    quad = safe_Et2 - 2.0 * safe_x * safe_Et + safe_x * safe_x
    ll_el_raw = -0.5 * (c2pi + torch.log(var) + quad / var)

    # Branchless: invalid entries contribute 0 to sum/mean (was previously a
    # host-sync `if not valid.any(): return nan`). When literally every entry
    # is invalid, sum=0 and mean returns NaN naturally via 0/0.
    valid_f = valid.to(ll_el_raw.dtype)
    ll_el_masked = ll_el_raw * valid_f

    if reduce == "sum":
        return ll_el_masked.sum()
    elif reduce == "mean":
        return ll_el_masked.sum() / valid_f.sum()
    elif reduce == "none":
        out = torch.full_like(x, float("nan"))
        out[valid] = ll_el_raw[valid]
        return out
    else:
        raise ValueError("reduce must be 'sum', 'mean', or 'none'")
