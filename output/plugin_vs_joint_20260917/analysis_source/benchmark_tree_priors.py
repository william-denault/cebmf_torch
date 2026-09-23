"""Reproducible tree-prior benchmark; never executes the supplied notebook.

The notebook scenario reproduces its data-generation code (including its two
indexing mistakes). The corrected scenario uses separate leaf sparsity masks
and assigns every row to a leaf. Truth is used only for simulation/evaluation.
Run from the repository root. Per-run JSON/checkpoints make runs resumable.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import time
from pathlib import Path

import torch

from cebmf_torch import cEBMF


class PluginCGB(cEBMF):
    """Benchmark-only old plug-in update: parents are fixed posterior means."""

    @property
    def _uses_conditional_inference(self):
        return False

    @property
    def _uses_joint_inference(self):
        return False


METHODS = {
    "gbinary": ("gbinary", False, cEBMF),
    "cgb": ("cgb", False, cEBMF),
    "cgb_plugin": ("cgb", True, PluginCGB),
    "cgb_self": ("cgb", True, cEBMF),
    "spiked_self": ("spiked_emdn", True, cEBMF),
    "spiked": ("spiked_emdn", False, cEBMF),
    "cgb_sharp": ("cgb_sharp", False, cEBMF),
    "cgb_sharp_plugin": ("cgb_sharp", True, PluginCGB),
    "cgb_sharp_self": ("cgb_sharp", True, cEBMF),
}


def simulate(seed, scenario="corrected", n=1000, p=200):
    if n % 4:
        raise ValueError("n must be divisible by four")
    g = torch.Generator().manual_seed(seed)
    masks = [torch.randint(0, 2, (p,), generator=g, dtype=torch.float32) for _ in range(7)]
    factors = torch.stack([
        masks[k if scenario == "corrected" or k < 3 else 3] * torch.randn(p, generator=g)
        for k in range(7)
    ])
    loadings = torch.zeros(n, 7)
    loadings[:, 0] = 1
    loadings[:n // 2, 1] = 1
    loadings[n // 2:, 2] = 1
    for leaf in range(4):
        start = leaf * n // 4
        if scenario == "notebook" and leaf > 0:
            start += 1
        loadings[start:(leaf + 1) * n // 4, leaf + 3] = 1
    signal = loadings @ factors
    # Preserve the notebook's float32 arithmetic and RNG sequence exactly.
    observed = signal + 0.5 * torch.randn(n, p, generator=g) * 2.5
    return observed, signal, loadings, factors.T


def rmse(a, b):
    return float((a.double() - b.double()).square().mean().sqrt())


def settings(args, method):
    prior, self_cov, cls = METHODS[method]
    row = ({"omega": args.omega} if prior == "gbinary" else dict(
        hidden_dim=args.hidden, n_layers=args.layers, n_epochs=args.epochs,
        batch_size=args.batch, lr=args.lr,
        penalty=(args.penalty if args.penalty is not None else
                 args.spiked_penalty if prior == "spiked_emdn" else args.cgb_penalty),
    ))
    if prior == "spiked_emdn":
        row["n_gaussians"] = args.components
    if prior == "cgb_sharp":
        row["omega"] = args.cgb_omega
    return cls, dict(
        K=args.rank, prior_L=prior, prior_F="norm", self_row_cov=self_cov,
        allow_backfitting=False, device="cpu", prior_L_kwargs=row,
        prior_F_kwargs={"penalty": args.ash_penalty},
        conditional_kwargs=dict(quadrature_points=args.quadrature,
                                parent_samples=args.parents, seed=args.integration_seed,
                                approximation=args.approximation),
    )


def metrics(model, signal, observed, test, elapsed, sweep):
    prediction = model.L @ model.F.T
    result = dict(sweep=sweep, seconds=elapsed, signal_rmse=rmse(prediction, signal),
                  train_observation_rmse=rmse(prediction[~test], observed[~test]),
                  sigma=float(model.tau.rsqrt().mean()), rank=model.model.K)
    if test.any():
        result.update(test_signal_rmse=rmse(prediction[test], signal[test]),
                      test_observation_rmse=rmse(prediction[test], observed[test]))
    if model.obj:
        result["objective"] = float(model.obj[-1])
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError(f"Nonfinite benchmark metric: {result}")
    return result


def run(args, seed, scenario, method):
    config = {k: v for k, v in vars(args).items() if k not in ("output", "seeds", "scenarios", "methods")}
    if config.get("orientation") == "raw":
        config.pop("orientation")  # Preserve identifiers of the original matched-SVD runs.
    config.update(seed=seed, scenario=scenario, method=method)
    _, resolved = settings(args, method)
    config.update(protocol_version=2, prior_L=resolved["prior_L"],
                  prior_L_kwargs=resolved["prior_L_kwargs"],
                  prior_F_kwargs=resolved["prior_F_kwargs"])
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    stem = f"{scenario}_{method}_seed{seed}_{args.initialization}_{signature}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.json"
    if path.exists() and json.loads(path.read_text())["complete"]:
        print(f"SKIP {stem}", flush=True)
        return
    observed, signal, true_l, true_f = simulate(seed, scenario, args.n, args.p)
    mask_g = torch.Generator().manual_seed(5000 + seed)
    test = torch.rand(observed.shape, generator=mask_g) < args.holdout
    train = observed.clone()
    train[test] = torch.nan
    torch.manual_seed(10000 + seed)
    initializer = cEBMF(train, K=args.rank, device="cpu", allow_backfitting=False)
    initializer.initialise_factors()
    initial_l, initial_f = initializer.L.clone(), initializer.F.clone()
    if args.orientation == "positive":
        # Resolve SVD's arbitrary column signs using only the observed-data fit.
        # This preserves L @ F.T exactly and is applied before either prior fit.
        signs = torch.where(initial_l.sum(0) >= 0, 1.0, -1.0)
        initial_l *= signs
        initial_f *= signs
    start = time.perf_counter()
    warm_history = []
    if args.initialization == "gbwarm":
        _, kw = settings(args, "gbinary")
        warm = cEBMF(train, **kw)
        warm.initialise_factors(L=initial_l, F=initial_f)
        for sweep in range(args.warm_steps):
            with contextlib.redirect_stdout(io.StringIO()):
                warm.iter_once()
            warm_history.append(metrics(warm, signal, observed, test, time.perf_counter() - start, sweep + 1))
        initial_l, initial_f = warm.L.clone(), warm.F.clone()
    warm_seconds = time.perf_counter() - start
    torch.manual_seed(10000 + seed)
    cls, kw = settings(args, method)
    model = cls(train, **kw)
    model.initialise_factors(L=initial_l.clone(), F=initial_f.clone())
    history = [metrics(model, signal, observed, test, 0.0, 0)]
    rank_l = int(torch.linalg.matrix_rank(true_l.double()))
    signal_sv = torch.linalg.svdvals(signal.double())
    record = dict(config=config, complete=False, data_rank=rank_l,
                  signal_singular_values=signal_sv[:10].tolist(),
                  initialization_rmse=history[0]["signal_rmse"], warm_seconds=warm_seconds,
                  warm_history=warm_history, history=history)
    start = time.perf_counter()
    for sweep in range(1, args.steps + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            model.iter_once()
        history.append(metrics(model, signal, observed, test, time.perf_counter() - start, sweep))
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if sweep == 1 or sweep % 5 == 0 or sweep == args.steps:
            print(f"{stem} sweep={sweep} signal={history[-1]['signal_rmse']:.5f} "
                  f"sigma={history[-1]['sigma']:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
    record["complete"] = True
    record["inference"] = (model._conditional_result().inference if model.conditional_fit else
                           "plugin" if method.endswith("_plugin") else "variational")
    if model.conditional_fit and args.approximation == "quadratic":
        history_local = model.conditional_fit.row.history
        record["curvature_clipped"] = float(sum(item["curvature_clipped"] for item in history_local))
        record["curvature_evaluations"] = sum(item["curvature_evaluations"] for item in history_local)
    if model.conditional_fit:
        graph = model.conditional_fit.row
        record["conditional_prior_parameters"] = [
            {key: value.tolist() for key, value in prior.state_dict().items()
             if key in ("log_slab_sd", "net.mu_2", "net.mu.bias", "net.log_sigma.bias")}
            for prior in graph.priors
        ]
    torch.save(dict(L=model.L, F=model.F, L2=model.L2, F2=model.F2,
                    truth_L=true_l, truth_F=true_f, test_mask=test), output / f"{stem}.pt")
    if model.conditional_fit:
        torch.save(model, output / f"{stem}.model.pt")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--scenarios", nargs="+", choices=["notebook", "corrected"], default=["notebook", "corrected"])
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=["gbinary", "cgb", "cgb_plugin", "cgb_self", "spiked_self"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--p", type=int, default=200)
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--penalty", type=float, default=None,
                   help="Optional shared override of both neural-family penalties (legacy control).")
    p.add_argument("--cgb-penalty", type=float, default=1.0511,
                   help="Spike penalty for CGB and sharp CGB, with or without self covariates.")
    p.add_argument("--spiked-penalty", type=float, default=1.1,
                   help="Spike penalty for spiked EMDN, with or without self covariates.")
    p.add_argument("--cgb-omega", type=float, default=0.01,
                   help="Sharp CGB only; plain CGB has no omega parameter.")
    p.add_argument("--ash-penalty", type=float, default=10.0)
    p.add_argument("--omega", type=float, default=0.1,
                   help="Ordinary gbinary scale ratio; separate from sharp CGB's omega.")
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--quadrature", type=int, default=16)
    p.add_argument("--approximation", choices=["quadrature", "quadratic"], default="quadratic",
                   help="Child feedback: quadratic (default, faster) or quadrature; both retain parent uncertainty.")
    p.add_argument("--parents", type=int, default=16)
    p.add_argument("--integration-seed", type=int, default=0)
    p.add_argument("--holdout", type=float, default=0.0)
    p.add_argument("--initialization", choices=["svd", "gbwarm"], default="svd")
    p.add_argument("--orientation", choices=["raw", "positive"], default="raw")
    p.add_argument("--warm-steps", type=int, default=10)
    p.add_argument("--output", default="output/tree_prior_benchmark/penalty_20260916/main")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    torch.set_num_threads(1)
    for scenario in args.scenarios:
        for seed in args.seeds:
            for method in args.methods:
                run(args, seed, scenario, method)
