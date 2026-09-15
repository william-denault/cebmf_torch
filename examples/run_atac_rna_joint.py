"""Run the reproducible joint-inference example from a terminal or notebook.

From the repository root: python examples/run_atac_rna_joint.py --quick
Without --quick, use the original N=2000, P=1000 simulation. Results are
diagnostics for an experimental finite-chain fit, not a comparative benchmark.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from cebmf_torch.experimental.conditional import SCALAR_PRIORS
from cebmf_torch.experimental.data import align_modalities, simulate_atac_rna
from cebmf_torch.experimental.joint import JointATACRNA


def run_example(
    *,
    n=2000,
    p=1000,
    seed=1,
    prior_atac="cgb",
    prior_rna="cgb",
    unpaired_fraction=0.2,
    use_side_info=False,
    rounds=8,
    burnin=100,
    draws=100,
    thin=2,
    holdout_fraction=0.02,
):
    """Return data, sampler and posterior; simulated truth is used only to score.

    unpaired_fraction is the fraction of all samples exclusive to *each*
    modality. At 0.2, 60% are paired, 20% ATAC-only and 20% RNA-only.
    RNA rows are deliberately reordered to exercise explicit ID alignment.
    """
    if not 0 <= unpaired_fraction < 0.5 or not 0 <= holdout_fraction < 0.5:
        raise ValueError("Fractions must lie in [0, 0.5).")
    torch.set_num_threads(1)
    sim = simulate_atac_rna(n, p, seed=seed, informative_side_info=use_side_info)
    exclusive = int(n * unpaired_fraction)
    ai, ri = torch.arange(n - exclusive), torch.arange(n - 1, exclusive - 1, -1)
    observed_a, observed_r = sim["atac"][ai].clone(), sim["rna"][ri].clone()
    generator = torch.Generator().manual_seed(seed + 2048)
    for value in (observed_a, observed_r):
        value[torch.rand(value.shape, generator=generator) < holdout_fraction] = torch.nan
    side = {"side_info": sim["side_info"], "side_ids": range(n)} if use_side_info else {}
    data = align_modalities(observed_a, observed_r, ai, ri, **side)
    started = time.perf_counter()
    print(f"Initializing {n} union rows, {p} features/view; priors {prior_atac}, {prior_rna}...", flush=True)
    solver = JointATACRNA(
        data,
        prior_atac=prior_atac,
        prior_rna=prior_rna,
        initial_hmm_kwargs={"half_grid": 8, "maxiter": 20},
        seed=seed + 100,
    )
    print(f"Retained ranks: ATAC {solver.ka}, RNA {solver.kr}. Learning loading priors...", flush=True)
    for i in range(rounds):
        solver.fit_prior_parameters(rounds=1, sweeps_per_round=10, steps=30)
        print(f"Prior-learning round {i + 1}/{rounds}", flush=True)
    learning_seconds = time.perf_counter() - started
    result = solver.sample(burnin=burnin, draws=draws, thin=thin, progress_every=25)
    row_ids = torch.tensor(data.ids, dtype=torch.long)
    metrics = []
    paired = data.atac_observed & data.rna_observed
    for m, (label, posterior) in enumerate((("atac", result.atac), ("rna", result.rna))):
        truth = sim[f"signal_{label}"][row_ids].double()
        observed_rows = data.atac_observed if m == 0 else data.rna_observed
        baseline = torch.full_like(truth, torch.nan)
        initial = solver.initial_models[m]
        baseline[observed_rows] = initial.L @ initial.F.T
        for group, mask in (
            ("paired", paired),
            ("observed_unpaired", observed_rows & ~paired),
            ("missing_modality", ~observed_rows),
        ):
            if mask.any():
                metrics.append(
                    {
                        "modality": label,
                        "rows": group,
                        "n": int(mask.sum()),
                        "joint_signal_mse": float((posterior.reconstruction[mask] - truth[mask]).square().mean()),
                        "independent_signal_mse": float((baseline[mask] - truth[mask]).square().mean())
                        if group != "missing_modality"
                        else None,
                    }
                )
        observed = data.atac if m == 0 else data.rna
        held_out = torch.isnan(observed) & observed_rows[:, None]
        if held_out.any():
            noisy = sim[label][row_ids].double()
            metrics.append(
                {
                    "modality": label,
                    "rows": "held_out_entries",
                    "n": int(held_out.sum()),
                    "joint_noisy_mse": float((posterior.reconstruction[held_out] - noisy[held_out]).square().mean()),
                    "independent_noisy_mse": float((baseline[held_out] - noisy[held_out]).square().mean()),
                }
            )
    report = {
        "config": {
            "n": n,
            "p": p,
            "seed": seed,
            "prior_atac": prior_atac,
            "prior_rna": prior_rna,
            "unpaired_fraction_per_view": unpaired_fraction,
            "use_side_info": use_side_info,
            "rounds": rounds,
            "burnin": burnin,
            "draws": draws,
            "thin": thin,
            "holdout_fraction": holdout_fraction,
        },
        "ranks": [solver.ka, solver.kr],
        "learning_seconds": learning_seconds,
        "total_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "acceptance": result.acceptance.tolist(),
        "log_joint": result.log_joint,
        "interpretation": (
            "Finite-chain diagnostic for one fitted conditional-EB target; not a convergence certificate or benchmark."
        ),
    }
    return sim, data, solver, result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Use 400 rows, 200 features and shorter learning/sampling."
    )
    parser.add_argument("--prior-atac", choices=SCALAR_PRIORS, default="cgb")
    parser.add_argument("--prior-rna", choices=SCALAR_PRIORS, default="cgb")
    parser.add_argument("--paired", action="store_true", help="Keep all samples paired.")
    parser.add_argument(
        "--side-info", action="store_true", help="Generate informative fixed covariates before the latent variables."
    )
    parser.add_argument("--output", default="output/joint_atac_rna_diagnostic.json")
    args = parser.parse_args()
    quick = {"n": 400, "p": 200, "rounds": 3, "burnin": 40, "draws": 40, "thin": 1} if args.quick else {}
    *_, report = run_example(
        prior_atac=args.prior_atac,
        prior_rna=args.prior_rna,
        unpaired_fraction=0 if args.paired else 0.2,
        use_side_info=args.side_info,
        **quick,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log_joint"}, indent=2))
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
