"""Tree simulation using automatic cEBMF joint inference.

Run: python examples/run_tree_joint.py --quick
Full size: 1,000 samples and 200 features, matching the notebook's effective N.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from cebmf_torch import cEBMF
from cebmf_torch.experimental.conditional import SCALAR_PRIORS


def simulate_tree(n=1000, p=200, *, seed=1, sigma=1.25):
    """Seven additive programs, four disjoint leaves; no gaps or reused masks."""
    if n < 20 or p < 10 or sigma <= 0:
        raise ValueError("Use n >= 20, p >= 10 and sigma > 0.")
    generator = torch.Generator().manual_seed(seed)
    mask = torch.randint(0, 2, (p, 7), generator=generator).double()
    f = mask * torch.randn(p, 7, generator=generator, dtype=torch.float64)
    l = torch.zeros(n, 7, dtype=torch.float64)
    c1, c2, c3 = int(0.1 * n), int(0.3 * n), int(0.8 * n)
    l[:, 0] = 1
    l[:c2, 1], l[c2:, 2] = 1, 1
    l[:c1, 3], l[c1:c2, 4], l[c2:c3, 5], l[c3:, 6] = 1, 1, 1, 1
    signal = l @ f.T
    observed = signal + sigma * torch.randn(n, p, generator=generator, dtype=torch.float64)
    return dict(observed=observed, signal=signal, L=l, F=f, leaf=l[:, 3:].argmax(1))


def run_example(*, n=1000, p=200, seed=1, prior_L="cgb", column_hierarchy=False,
                rounds=8, burnin=100, draws=150, thin=2, initialization_iterations=10, pretrain_steps=100):
    torch.set_num_threads(1)
    sim = simulate_tree(n, p, seed=seed)
    heldout = torch.rand(n, p, generator=torch.Generator().manual_seed(seed + 400)) < 0.02
    data = sim["observed"].clone()
    data[heldout] = torch.nan
    start = time.perf_counter()
    baseline = cEBMF(data, K=10, prior_L="norm", prior_F="norm", S=1.25, device="cpu")
    baseline.initialise_factors()
    baseline.fit(initialization_iterations)
    model = cEBMF(
        data, K=10, prior_L=prior_L, prior_F="spiked_emdn" if column_hierarchy else "norm",
        self_row_cov=True, self_col_cov=column_hierarchy, S=1.25, device="cpu",
        joint_kwargs=dict(initialization_iterations=initialization_iterations, pretrain_steps=pretrain_steps,
                          seed=seed + 100, burnin=burnin, draws=draws, thin=thin, progress_every=25),
    )
    model.initialise_factors()
    result = model.fit(rounds)
    independent = baseline.L @ baseline.F.T
    report = dict(
        n=n, p=p, seed=seed, prior_L=prior_L, column_hierarchy=column_hierarchy,
        rank=model.model.K, rounds=rounds, burnin=burnin, draws=draws, thin=thin,
        signal_rmse_joint=float((result.reconstruction - sim["signal"]).square().mean().sqrt()),
        signal_rmse_baseline=float((independent - sim["signal"]).square().mean().sqrt()),
        heldout_rmse_joint=float((result.reconstruction[heldout] - sim["observed"][heldout]).square().mean().sqrt()),
        heldout_rmse_baseline=float((independent[heldout] - sim["observed"][heldout]).square().mean().sqrt()),
        elapsed_seconds=time.perf_counter() - start,
        interpretation="Finite-chain diagnostic. Seven generating programs yield four leaf profiles; reconstruction "
                       "does not identify the generating tree. The baseline also differs in prior family.",
    )
    return sim, model, result, baseline, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prior", choices=SCALAR_PRIORS, default="cgb")
    parser.add_argument("--column-hierarchy", action="store_true")
    parser.add_argument("--output", default="output/tree_joint_diagnostic.json")
    args = parser.parse_args()
    settings = dict(n=300, p=80, rounds=3, burnin=30, draws=40, thin=1,
                    initialization_iterations=5, pretrain_steps=50) if args.quick else {}
    *_, report = run_example(prior_L=args.prior, column_hierarchy=args.column_hierarchy, **settings)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
