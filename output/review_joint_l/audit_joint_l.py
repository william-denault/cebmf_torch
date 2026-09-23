"""Read-only audit experiments: ash F, joint L, and the legacy plug-in update.

No production source is modified. Run from the repository root with the
cebmf Python environment. JSON outputs distinguish evidence from hypotheses.
"""

import argparse
import contextlib
import copy
import io
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "archive" / "tree"))

from run_tree_joint import simulate_tree
from cebmf_torch import cEBMF
from cebmf_torch.experimental.conditional import GaussianMixture, metropolis_normal_means
from cebmf_torch.experimental.matrix import JointMatrix


class LegacyPlugin(cEBMF):
    """Exercise retained old moment updates with earlier-loading covariates."""

    @property
    def _uses_joint_inference(self):
        return False

    @property
    def _uses_conditional_inference(self):
        return False


def rmse(x, y):
    return float((x - y).square().mean().sqrt())


def setup_checks():
    g = torch.Generator().manual_seed(7)
    y = torch.randn(30, 20, generator=g, dtype=torch.float64)
    l = torch.rand(30, 2, generator=g, dtype=torch.float64) * 0.4
    f = torch.randn(20, 2, generator=g, dtype=torch.float64)
    records = []
    for factor_kwargs in ({"penalty": 1.0}, {"penalty": 80.0, "mode": 3.0}):
        owner = cEBMF(y, K=2, prior_L="spiked_emdn", prior_F="norm", S=1.25,
                      self_row_cov=True, device="cpu", prior_F_kwargs=factor_kwargs,
                      joint_kwargs={"pretrain_steps": 0, "progress_every": 0})
        owner.initialise_factors(L=l.clone(), F=f.clone())
        # Observe constructor state before its first stochastic sweep.
        with patch.object(JointMatrix, "sweep", return_value=None):
            engine = JointMatrix(owner)
        records.append({"nonzero_L_before": int((l != 0).sum()),
                        "nonzero_L_after_constructor": int((engine.values[0] != 0).sum()),
                        "factor_prior_mean": engine.axes[1].fixed[0].mean.tolist(),
                        "factor_prior_log_weight": engine.axes[1].fixed[0].log_weight.tolist()})
    # Repeat the ignored-settings check with informative, nonzero loading
    # columns, so it does not depend on the all-zero fallback above.
    ash_records = []
    for factor_kwargs in ({"penalty": 1.0}, {"penalty": 80.0, "mode": 3.0}):
        owner = cEBMF(y, K=2, prior_L="spiked_emdn", prior_F="norm", S=1.25,
                      self_row_cov=True, device="cpu", prior_F_kwargs=factor_kwargs,
                      joint_kwargs={"pretrain_steps": 0, "progress_every": 0})
        owner.initialise_factors(L=l.clone()+1, F=f.clone())
        with patch.object(JointMatrix, "sweep", return_value=None):
            engine = JointMatrix(owner)
        ash_records.append([(mix.mean.tolist(), mix.log_weight.tolist(), mix.variance.tolist())
                            for mix in engine.axes[1].fixed])
    from cebmf_torch.cebnm.cov_gb_prior import m_step_sigma2
    x, se = torch.tensor([2., 0.]), torch.tensor([.1, 10.])
    updated = m_step_sigma2(torch.ones_like(x), torch.tensor(0.), x, se)
    def slab_loglik(variance):
        return float(torch.distributions.Normal(0., (se.square()+variance).sqrt()).log_prob(x).sum())
    return {"threshold_probe": {key: records[0][key] for key in
                                ("nonzero_L_before", "nonzero_L_after_constructor")},
            "ash_kwargs_have_no_effect_nonzero_loadings": ash_records[0] == ash_records[1],
            "legacy_cgb_heteroscedastic_variance_counterexample": {
                "x":x.tolist(), "se":se.tolist(), "old_variance":4., "new_variance":float(updated),
                "loglik_before":slab_loglik(4.), "loglik_after":slab_loglik(updated)}}


def metastability_check():
    """Exactly enumerable binary hierarchy: valid kernel, misleading acceptance."""
    n, epsilon, sweeps = 400, 1e-6, 1000
    dtype = torch.float64
    generator = torch.Generator().manual_seed(24)
    # Two true atoms. This is a transparent limiting example of sharp priors,
    # not a claim that production CGB has an atom at one.
    prior = GaussianMixture(torch.full((n, 2), -math.log(2), dtype=dtype),
                            torch.tensor([0., 1.], dtype=dtype).expand(n, -1),
                            torch.zeros(n, 2, dtype=dtype))
    zero = torch.zeros(n, dtype=dtype)
    parent, child = zero.clone(), zero.clone()
    label = torch.zeros(n, dtype=torch.long)
    accepts, moves = 0., 0.
    for _ in range(sweeps):
        old = parent.clone()
        def child_log(value):
            return torch.where(value == child, math.log1p(-epsilon), math.log(epsilon))
        parent, label, accept = metropolis_normal_means(prior, zero, zero, parent, label, child_log, generator)
        accepts += float(accept.double().mean())
        moves += float((parent != old).double().mean())
        child = torch.where(torch.rand(n, generator=generator) < epsilon, 1 - parent, parent)
    return {"exact_parent_probability_one": 0.5, "empirical_probability_one": float(parent.mean()),
            "parent_acceptance": accepts / sweeps, "parent_actual_move_fraction": moves / sweeps,
            "child_acceptance": 1.0, "epsilon": epsilon, "parallel_chains": n, "sweeps": sweeps}


@torch.no_grad()
def collect(engine, seed, burnin, draws):
    engine = copy.deepcopy(engine)
    engine.generator.manual_seed(seed)
    # deepcopy preserves shared generator references within this engine.
    for _ in range(burnin):
        engine.sweep()
    loading, factor, accepts, changes = [], [], [], []
    for _ in range(draws):
        old = engine.axes[0].components.clone()
        accepts.append(engine.sweep()[0])
        changes.append((old != engine.axes[0].components).double().mean(0))
        loading.append(engine.values[0].clone())
        factor.append(engine.values[1].clone())
    l, f = torch.stack(loading), torch.stack(factor)
    return l, f, torch.stack(accepts).mean(0), torch.stack(changes).mean(0)


def split_rhat(x):
    # Classical split Rhat on signal coordinates; a diagnostic, not a certificate.
    half = x.shape[1] // 2
    x = torch.cat((x[:, :half], x[:, -half:]), 0)
    within = x.var(1, unbiased=True).mean(0)
    between = half * x.mean(1).var(0, unbiased=True)
    return (((half - 1) / half * within + between / half) / within.clamp_min(1e-20)).sqrt()


def benchmark(seed, prior, burnin, draws, rounds, n, p):
    start = time.perf_counter()
    sim = simulate_tree(n, p, seed=seed)
    gen = torch.Generator().manual_seed(seed + 700)
    heldout = torch.rand(n, p, generator=gen) < .05
    data = sim["observed"].clone()
    data[heldout] = torch.nan
    common = dict(data=data, K=4, prior_F="norm", S=1.25, device="cpu", allow_backfitting=False,
                  prior_F_kwargs={"penalty": 1.0})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        base = cEBMF(**common, prior_L="norm", prior_L_kwargs={"penalty": 1.0})
        base.initialise_factors()
        base.fit(20)
    l, f = base.L.clone(), base.F.clone()
    scale = l.square().sum(0) / l.abs().sum(0).clamp_min(1e-12)
    l, f = l / scale, f * scale
    settings = dict(penalty=1.0, n_epochs=20, hidden_dim=16, n_layers=1, batch_size=256, lr=.003)
    kwargs = dict(**common, prior_L=prior, prior_L_kwargs=settings, self_row_cov=True)
    torch.manual_seed(seed + 200)
    legacy = LegacyPlugin(**kwargs)
    legacy.initialise_factors(L=l.clone(), F=f.clone())
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        legacy.fit(20)
    joint = cEBMF(**kwargs, joint_kwargs=dict(seed=seed+200, sweeps_per_round=10,
                  elbo_draws=2, progress_every=0, initialization_iterations=0))
    joint.initialise_factors(L=l.clone(), F=f.clone())
    joint._ensure_joint_sampler()
    engine = joint.joint_sampler
    # ELBO diagnostics do not control learning. Omit their runtime in the audit.
    engine.record_elbo = lambda *args, **kwargs: None
    frozen_f = [mix.log_weight.clone() for mix in engine.axes[1].fixed]
    engine.fit_prior_parameters(rounds)
    factor_unchanged = all(torch.equal(w, mix.log_weight) for w, mix in zip(frozen_f, engine.axes[1].fixed))
    references = {"independent_ash": base.L @ base.F.T, "legacy_plugin": legacy.L @ legacy.F.T}
    result = {"seed": seed, "prior_L": prior, "n": n, "p": p, "rank":4,
              "known_noise_sd":1.25, "all_penalties":1.0, "rounds":rounds,
              "ash_parameters_unchanged_during_learning":factor_unchanged,
              "reference_signal_rmse":{k:rmse(v, sim["signal"]) for k,v in references.items()},
              "reference_heldout_rmse":{k:rmse(v[heldout],sim["observed"][heldout]) for k,v in references.items()},
              "chains":[], "scope":"Matched warm factors, rank, noise, architecture, explicit penalty. Legacy and joint still differ in optimizer objective, scales, initialization and F learning."}
    means, monitor, first_halves, last_halves = [], [], [], []
    indices = torch.randint(n*p, (24,), generator=gen)
    for chain in range(3):
        loading, factor, accept, switch = collect(engine, seed+1000+chain, burnin, draws)
        signal = loading @ factor.transpose(1,2)
        mean = signal.mean(0)
        means.append(mean)
        first_halves.append(signal[:draws//2].mean(0))
        last_halves.append(signal[draws//2:].mean(0))
        monitor.append(signal.flatten(1)[:, indices])
        result["chains"].append({"seed":seed+1000+chain,"burnin":burnin,"draws":draws,
                    "signal_rmse":rmse(mean,sim["signal"]),
                    "heldout_rmse":rmse(mean[heldout],sim["observed"][heldout]),
                    "rmse_product_of_marginal_means":rmse(loading.mean(0)@factor.mean(0).T,sim["signal"]),
                    "L_acceptance":accept.tolist(),"L_component_switch_fraction":switch.tolist(),
                    "prefix_signal_rmse":{str(d):rmse(signal[:d].mean(0),sim["signal"]) for d in (min(50,draws),min(200,draws),draws)},
                    "first_last_half_signal_difference_rmse":rmse(first_halves[-1],last_halves[-1])})
    pooled = torch.stack(means).mean(0)
    result.update(pooled_signal_rmse=rmse(pooled,sim["signal"]),
                  pooled_heldout_rmse=rmse(pooled[heldout],sim["observed"][heldout]),
                  chain_mean_disagreement_rmse=rmse(torch.stack(means),pooled),
                  selected_signal_split_rhat_max=float(split_rhat(torch.stack(monitor)).max()),
                  elapsed_seconds=time.perf_counter()-start)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prior",default="spiked_emdn")
    parser.add_argument("--seeds",type=int,nargs="+",default=[1,2,3])
    parser.add_argument("--burnin",type=int,default=200)
    parser.add_argument("--draws",type=int,default=600)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--n",type=int,default=240)
    parser.add_argument("--p",type=int,default=80)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result={"setup":setup_checks(),"metastability":metastability_check(),"benchmarks":[]}
    path=args.output or Path(__file__).with_name(f"results_{args.prior}.json")
    for seed in args.seeds:
        record=benchmark(seed,args.prior,args.burnin,args.draws,args.rounds,args.n,args.p)
        result["benchmarks"].append(record)
        path.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(record),flush=True)
    print(f"Saved {path}",flush=True)


if __name__ == "__main__":
    main()
