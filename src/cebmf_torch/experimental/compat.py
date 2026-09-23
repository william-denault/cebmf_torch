"""Compatibility with the deprecated explicitly requested MCMC backend."""

import torch

from cebmf_torch.cebmf.cebmf import cEBMF, NoiseType


@torch.no_grad()
def ensure_joint_sampler(owner):
    from cebmf_torch.experimental.matrix import HMM_PRIORS, JointMatrix, joint_options

    if owner.joint_sampler is not None:
        owner.joint_sampler.check_fixed_inputs(owner)
        return
    options = joint_options(owner.joint_kwargs)
    JointMatrix.validate_owner(owner)
    if not owner._factors_initialised:
        owner.initialise_factors()
    # The preliminary independent fit chooses a starting rank and noise.
    # User-supplied fitted factors keep their rank, values and ordering.
    if not owner._user_initialisation and options["initialization_iterations"]:
        names = (owner.model.prior_L, owner.model.prior_F)
        initial_names = [name if name in HMM_PRIORS else "norm" for name in names]
        for m, name in enumerate(names):
            if name in ("cgb", "cgb_sharp", "cgb_sharp_2"):
                initial_names[m] = "gbinary"
        with torch.random.fork_rng():
            torch.manual_seed(options["seed"])
            source = cEBMF(
                owner.Y, K=owner.model.K, prior_L=initial_names[0], prior_F=initial_names[1],
                prior_L_kwargs=owner._prior_L_kwargs if names[0] in HMM_PRIORS else {},
                prior_F_kwargs=owner._prior_F_kwargs if names[1] in HMM_PRIORS else {},
                S=owner._S_input,
                noise_type=NoiseType.CONSTANT if owner._S_input is not None else owner.noise.type,
                allow_backfitting=owner.model.allow_backfitting, prune_thresh=owner.model.prune_thresh,
                device="cpu", verbose=False,
            )
            source.initialise_factors()
            source.fit(options["initialization_iterations"])
        if source.model.K < 1:
            raise ValueError("Independent initialization retained no factors; supply fitted initial L and F.")
        owner.model.K = source.model.K
        owner.L, owner.F, owner.L2, owner.F2 = (getattr(source, key).double().clone() for key in ("L", "F", "L2", "F2"))
        owner.tau = source.tau.clone()
        owner.tau_map = source.tau_map.clone()
        owner.model_state_L, owner.model_state_F = source.model_state_L, source.model_state_F
        owner.kl_l, owner.kl_f = torch.zeros(owner.model.K), torch.zeros(owner.model.K)
        owner.pi0_L, owner.pi0_F = source.pi0_L, source.pi0_F
        # Fix a useful loading scale before learning conditional priors.
        # Preserve L F.T and transform any opposite-axis HMM parameters.
        axis = next((m for m, name in enumerate(names) if name in ("cgb", "cgb_sharp", "cgb_sharp_2")), None)
        if axis is not None:
            value, other = (owner.L, owner.F) if axis == 0 else (owner.F, owner.L)
            scale = value.square().sum(0) / value.abs().sum(0).clamp_min(1e-12)
            scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
            value.div_(scale)
            other.mul_(scale)
            states = owner.model_state_F if axis == 0 else owner.model_state_L
            if names[1 - axis] in HMM_PRIORS:
                for k, state in enumerate(states):
                    state["mu"] = state["mu"] * scale[k]
                    state["prior_sd"] = state["prior_sd"] * scale[k]
            owner.L2, owner.F2 = owner.L.square(), owner.F.square()
    owner.Y0 = owner.Y0.double()
    owner.mask = owner.mask.double()
    owner.L, owner.F = owner.L.double(), owner.F.double()
    owner.L2, owner.F2 = owner.L2.double(), owner.F2.double()
    owner.obj = []
    owner.joint_sampler = JointMatrix(owner)

