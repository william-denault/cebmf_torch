"""Original scalar plug-in cEBMF with selectable parent-moment covariates.

This benchmark changes neural inputs only. It adds neither parent integration
nor child-prior feedback, and does not change the public cEBMF defaults.
"""

import math

import torch

from benchmark_tree_priors import PluginCGB


class MomentPluginCEBMF(PluginCGB):
    def __init__(self, *args, row_cov_moments='mean', **kwargs):
        if row_cov_moments not in ('mean', 'mean_second'):
            raise ValueError("row_cov_moments must be 'mean' or 'mean_second'")
        self.row_cov_moments = row_cov_moments
        super().__init__(*args, **kwargs)
        self.fitted_row_inputs = [None] * self.model.K

    @torch.no_grad()
    def row_inputs(self, k):
        inputs = super()._build_covariate_matrix(
            self.covariate.X_l, self.covariate.self_row_cov, self.L, k, self.N)
        if self.covariate.self_row_cov and k and self.row_cov_moments == 'mean_second':
            inputs = torch.cat((inputs, self.L2[:, :k]), dim=1)
        return inputs

    @torch.no_grad()
    def _build_covariate_matrix(self, external_cov, self_cov_enabled, factors, k, dim_size):
        if factors is self.L:
            inputs = self.row_inputs(k)
            self.fitted_row_inputs[k] = None if inputs is None else inputs.detach().clone()
            return inputs
        return super()._build_covariate_matrix(external_cov, self_cov_enabled, factors, k, dim_size)

    @torch.no_grad()
    def _prune_indices(self, idxs):
        keep = [k for k in range(self.model.K) if k not in idxs]
        super()._prune_indices(idxs)
        if idxs:
            self.fitted_row_inputs = [self.fitted_row_inputs[k] for k in keep]


@torch.no_grad()
def frozen_elbo_terms(model):
    """Evaluate the fitted per-row-prior ELBO, not an autoregressive joint ELBO.

Cached KLs use the normal-means evidence identity at the last fitted prior.
Later likelihood/noise changes do not invalidate a KL between that unchanged
q and prior. Pruning can make retained input snapshots stale until the next
sweep; this is reported explicitly instead of silently relabeling the score.
The expected Gaussian likelihood is recomputed in float64 on the same device.
"""
    l, f = model.L.double(), model.F.double()
    vl, vf = model.L2.double() - l.square(), model.F2.double() - f.square()
    expected_error = ((model.Y0.double() - l @ f.T).square()
                      + vl @ f.square().T + l.square() @ vf.T + vl @ vf.T)
    mask = model.mask.bool()
    tau = model.tau_map.double()
    ll = -.5 * torch.where(mask, math.log(2 * math.pi) - tau.log() + tau * expected_error, 0.).sum()
    kl_l, kl_f = model.kl_l.double().sum(), model.kl_f.double().sum()
    elbo = ll - kl_l - kl_f
    penalty = float(model.prior_L_fn.kwargs.get('penalty', 1.))
    row_regularization = (penalty - 1) * sum(
        (p.double().clamp_min(1e-8).log().sum() for p in model.pi0_L), l.new_zeros(()))
    column_regularization = model.regularization_F.double().sum()
    matches = all(old is not None and old.shape == model.row_inputs(k).shape
                  and torch.equal(old, model.row_inputs(k))
                  for k, old in enumerate(model.fitted_row_inputs))
    return dict(elbo=float(elbo), expected_log_likelihood=float(ll),
                kl_loading=float(kl_l), kl_feature=float(kl_f),
                row_regularization=float(row_regularization),
                column_regularization=float(column_regularization),
                penalized_elbo=float(elbo + row_regularization + column_regularization),
                package_negative_elbo=float(model.obj[-1]),
                package_score_discrepancy=float(elbo + model.obj[-1].double()),
                min_loading_kl=float(model.kl_l.min()), min_feature_kl=float(model.kl_f.min()),
                current_covariates_match_fitted=matches)
