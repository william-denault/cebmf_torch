"""Fit two cEBMF observation models with p(L_second | L_first)."""

import torch

from ._conditional import ConditionalFit


def fit_joint(first, second, maxit=50):
    """Jointly fit two cEBMF objects whose rows share the same cell-ID order.

    The directed prior is p(L_first) p(L_second | L_first); each object's
    self_row_cov controls dependencies *within* its own loadings. Feature
    priors and noise are updated by the ordinary cEBMF routines. Return two
    CEBMFResult objects; their histories contain the same joint objective.

    Use align_modalities before constructing the models for partially paired
    data. Missing observations are NaN, never zero. Both loading priors must
    be supported conditional Gaussian mixtures. Rank is fixed during a
    coupled fit because deleting a column changes the prior graph.

    Print one completion message per joint sweep if either model has
    verbose=True (the default). Set verbose=False on both models to silence
    sweep progress. Warnings remain enabled.
    """
    if isinstance(maxit, bool) or not isinstance(maxit, int) or maxit < 0:
        raise ValueError("maxit must be a nonnegative integer.")
    if first is second:
        raise ValueError("Supply two distinct cEBMF models.")
    if first.N != second.N:
        raise ValueError("Align rows by cell ID with align_modalities before constructing the models.")
    if first.Y.device != second.Y.device or first.Y.dtype != second.Y.dtype:
        raise ValueError("Coupled models must use the same device and floating-point dtype.")
    if first.joint_kwargs or second.joint_kwargs or first.joint_sampler or second.joint_sampler:
        raise ValueError("fit_joint uses variational inference; remove legacy joint_kwargs.")
    if first.conditional_kwargs != second.conditional_kwargs:
        raise ValueError("Use the same conditional_kwargs for both models.")
    present_a, present_r = first.mask.bool().any(1), second.mask.bool().any(1)
    if not (present_a | present_r).all():
        raise ValueError("Each aligned cell must have measurements in at least one modality.")
    if not (present_a & present_r).any():
        raise ValueError("Learning an unrestricted cross-modality prior requires paired cells.")
    for model in (first, second):
        if not model._factors_initialised:
            model.initialise_factors()
    engine = first.conditional_fit
    if engine is None or engine is not second.conditional_fit:
        if any(m.conditional_fit is not None for m in (first, second)):
            raise ValueError("Initialize fresh model objects (or reset their factors) before coupling different graphs.")
        with torch.no_grad():
            engine = ConditionalFit([first, second], first.conditional_kwargs, cross=True)
        first.conditional_fit = second.conditional_fit = engine
    elif engine.models != [first, second]:
        raise ValueError("Keep the same modality order when continuing fit_joint.")
    for _ in range(maxit):
        engine.step()
    engine.finish()
    return first._conditional_result(), second._conditional_result()
