"""Compare one loading update with fixed versus uncertain earlier loadings.

Every method starts from the same stored factors, moments, noise precision and
own neural weights. Fixed side information is a frozen copy of L[:, :k], so its
dimension AND values match the self-covariate means. Setup/reset and the first
warm-up update are excluded. This measures runtime, not reconstruction quality.
"""

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import cebmf_torch
from cebmf_torch import cEBMF

from benchmark_tree_priors import simulate


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def settings(args, family):
    row = dict(penalty=1.1 if family == "spiked_emdn" else 1.0511,
               hidden_dim=64 if family == "spiked_emdn" else 32,
               n_layers=4, n_epochs=10, batch_size=128, lr=0.001)
    numerical = dict(approximation="quadratic", quadrature_points=24, parent_samples=32)
    if args.preset == "compact":
        row.update(hidden_dim=16, n_layers=0, n_epochs=2, batch_size=args.n, lr=0.01)
        numerical.update(quadrature_points=8, parent_samples=8)
    if family == "spiked_emdn":
        row["n_gaussians"] = 5
    if family == "cgb_sharp_2":
        row["omega"] = 0.01
    return row, numerical


def make_case(template, method, k, row):
    """Only benchmark controls use private APIs; the public API is unchanged."""
    if method == "fixed":
        frozen = template.L[:, :k].clone()
        model = cEBMF(template.Y, K=template.model.K, device=template.L.device,
                      prior_L=template.model.prior_L, prior_F="norm", X_l=frozen,
                      self_row_cov=False, allow_backfitting=False, prior_L_kwargs=row, verbose=False)
        model.initialise_factors(L=template.L.clone(), F=template.F.clone())
        model.L2.copy_(template.L2)
        model.F2.copy_(template.F2)
        model.tau = template.tau.clone()
        model.model_state_L[k] = copy.deepcopy(template.conditional_fit.row.priors[k].net.state_dict())

        def update():
            model._recompute_residual()
            model._update_L_factor(k, model._partial_residual_masked(k), None, 1e-12)
        return model, update

    model = copy.deepcopy(template)
    graph = model.conditional_fit.row
    graph.options["approximation"] = "quadrature" if method == "quadrature" else "quadratic"
    if method == "fixed_profile":
        # Diagnostic control: the SAME conditional objective/optimizer, but with
        # point-valued inputs and no child feedback. Not a new public fit mode.
        graph.external[k] = template.L[:, :k].clone()
        graph.parents[k] = []
        graph.children[k] = []
    return model, lambda: graph.update(k)


def work_counts(model, update, k):
    """Count network input rows in an untimed pass, including repeated epochs."""
    graph = model.conditional_fit.row if model.conditional_fit else None
    if graph is None:
        update()
        return None
    counts = defaultdict(lambda: dict(calls=0, input_rows=0))
    handles = []
    for node, prior in enumerate(graph.priors):
        def hook(module, inputs, node=node):
            key = f'{"own" if node == k else "child"}_{inputs[0].dtype}'
            counts[key]["calls"] += 1
            counts[key]["input_rows"] += len(inputs[0])
        handles.append(prior.register_forward_pre_hook(hook))
    try:
        update()
    finally:
        for handle in handles:
            handle.remove()
    return dict(counts)


def run_case(template, family, method, k, args, row):
    device = template.L.device
    torch.manual_seed(20000 + args.seed)
    model, update = make_case(template, method, k, row)
    counts = work_counts(model, update, k)  # warm-up, not timed
    synchronize(device)
    assert torch.isfinite(model.L).all() and torch.isfinite(model.L2).all()
    del update, model
    times, peaks = [], []
    for repeat in range(args.repeats):
        torch.manual_seed(20000 + args.seed)
        model, update = make_case(template, method, k, row)
        synchronize(device)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        update()
        synchronize(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peaks.append((torch.cuda.max_memory_allocated(device) - baseline) / 2**20)
        assert model.L.device == device
        assert torch.isfinite(model.L).all() and torch.isfinite(model.L2).all()
        del update, model
        print(f'{family:12s} k={k} d={k} {method:13s} {repeat + 1}/{args.repeats}: '
              f'{times[-1]:.4f}s', flush=True)
    return dict(family=family, coordinate_zero_based=k, covariate_dimension=k,
                children=args.rank - k - 1 if method in ("quadratic", "quadrature") else 0,
                method=method, seconds=times, median_seconds=statistics.median(times),
                peak_extra_allocated_mib=peaks, network_work=counts)


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        # Fail immediately if only enumeration works, or execution is sandboxed.
        torch.ones(2, device=device).square()
        synchronize(device)
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(1)
    y, _, _, _ = simulate(args.seed, "notebook", args.n, args.p)
    initializer = cEBMF(y, K=args.rank, device="cpu", allow_backfitting=False)
    initializer.initialise_factors()
    initial_l, initial_f = initializer.L.to(device), initializer.F.to(device)
    y = y.to(device)
    root = Path(__file__).resolve().parents[3]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / "src/cebmf_torch").rglob("*.py")}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(python=sys.executable, torch=torch.__version__, hip=torch.version.hip,
                  device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  package=cebmf_torch.__file__, source_sha256=hashes,
                  configuration=vars(args), resolved_settings={}, results=[])
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "results")}, indent=2), flush=True)
    for family in args.families:
        row, numerical = settings(args, family)
        report["resolved_settings"][family] = dict(row=row, integration=numerical)
        torch.manual_seed(10000 + args.seed)
        template = cEBMF(y, K=args.rank, device=device, prior_L=family, prior_F="norm",
                         self_row_cov=True, allow_backfitting=False, prior_L_kwargs=row,
                         conditional_kwargs=numerical, verbose=False)
        template.initialise_factors(L=initial_l.clone(), F=initial_f.clone())
        template.fit(0)
        for k in args.coordinates:
            for method in args.methods:
                report["results"].append(run_case(template, family, method, k, args, row))
                (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del template
    lines = ["# Fixed versus self covariates: loading-update timings", "",
             f"Device: {report['gpu'] or 'CPU'}; torch {report['torch']}; preset `{args.preset}`.", "",
             "One coordinate update, not one full sweep. Setup, state reset and warm-up excluded.",
             "All repeats start from identical factors, moments, precision and own neural weights.",
             "Fixed inputs are a frozen copy of the earlier loading means, with the same dimension.", "",
             "| Prior | k (zero based) / input dimension | Method | Median seconds | Extra peak MiB |",
             "|---|---:|---|---:|---:|"]
    for r in report["results"]:
        peak = max(r["peak_extra_allocated_mib"], default=0)
        lines.append(f"| {r['family']} | {r['covariate_dimension']} | {r['method']} | {r['median_seconds']:.4f} | {peak:.1f} |")
    lines += ["", "`fixed` is the ordinary cEBMF row update with X_l and self_row_cov=False.",
              "`fixed_profile` is a diagnostic using the conditional optimizer with frozen inputs and no children.",
              "`quadratic` and `quadrature` integrate uncertain parents and include child feedback.",
              "The ordinary solver differs in optimizer/scaling details; in particular sharp-family omega",
              "shrinks the ordinary EM variance repeatedly but only initializes the joint learned variance.",
              "Thus this is a compute comparison, not evidence about estimator accuracy.",
              "Peak memory is extra allocated memory above each case's initial state, excluding allocator reserve.",
              "No column-factor updates, noise updates or full-model objective evaluations are timed.",
              "See timings.json for raw repeats, resolved settings, input-row counts and source hashes.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--preset", choices=["original", "compact"], default="original")
    parser.add_argument("--families", nargs="+", choices=["cgb", "spiked_emdn", "cgb_sharp_2"],
                        default=["cgb", "spiked_emdn"])
    parser.add_argument("--methods", nargs="+", choices=["fixed", "fixed_profile", "quadratic", "quadrature"],
                        default=["fixed", "quadratic", "quadrature"])
    parser.add_argument("--coordinates", type=int, nargs="+", default=[3])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--output", default="output/tree_covariate_update")
    args = parser.parse_args()
    if args.repeats < 1 or args.n < 4 or args.n % 4 or not all(0 < k < args.rank for k in args.coordinates):
        parser.error("Require repeats >= 1, n >= 4 divisible by 4, and 0 < each coordinate < rank.")
    main(args)
