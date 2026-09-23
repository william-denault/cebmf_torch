"""Time the user's three-stage tree example with explicit device and fit settings.

Default: one seed, a compact network and 3/2/3 sweeps, quadratic feedback.
--preset original retains the package's neural/integration defaults; it still
uses quadratic feedback unless --approximation quadrature is explicitly set.
The notebook simulation is preserved by default, including its indexing bugs.
Use --scenario corrected to fix those separately from the runtime comparison.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import cebmf_torch
from cebmf_torch import cEBMF

from benchmark_tree_priors import simulate


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_stage(y, truth, prior, steps, args, seed, initial=None):
    device = y.device
    # Reset neural initialization independently of the preceding stage's RNG use.
    torch.manual_seed(10000 + seed)
    row = {"penalty": 1.1 if prior == "spiked_emdn" else 1.051 if prior == "cgb" else 1.0511}
    if prior == "cgb_sharp_2":
        row["omega"] = 0.01
    numerical = {"approximation": args.approximation}
    if args.preset == "compact":
        # Explicitly a smaller training/capacity budget, not a lossless speedup.
        row.update(hidden_dim=16, n_layers=0, n_epochs=2, batch_size=len(y), lr=0.01)
        numerical.update(quadrature_points=8, parent_samples=8)
    # Leave n_gaussians at 5: compact mode preserves the user's four EMDN slabs.
    synchronize(device)
    start = time.perf_counter()
    m = cEBMF(y, K=args.rank, device=device, self_row_cov=True,
              prior_L=prior, prior_F="norm", allow_backfitting=False,
              prior_L_kwargs=row, conditional_kwargs=numerical, verbose=False)
    if initial is None:
        m.initialise_factors()
    else:
        # Initializes second moments, residuals and noise consistently as well.
        m.initialise_factors(L=initial.L.clone(), F=initial.F.clone())
    result = m.fit(0)  # Include graph construction in the separately reported setup.
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    graph = m.conditional_fit.row
    print(f"  {prior}: {result.inference}, device={m.L.device}, "
          f"epochs={graph.training[0]['n_epochs']}, "
          f"batches/factor={len(graph.batches[0])}, setup={setup_seconds:.2f}s", flush=True)
    times = []
    for step in range(1, steps + 1):
        start = time.perf_counter()
        m.iter_once()
        # Timing/reporting boundary only: there are no scalar reads in an epoch.
        synchronize(device)
        times.append(time.perf_counter() - start)
        if step == 1 or step == steps or step % 5 == 0:
            print(f"    sweep {step}/{steps}: {times[-1]:.3f}s", flush=True)
    rmse = (truth - m.L @ m.F.T).square().mean().sqrt().item()
    record = dict(prior=prior, inference=result.inference, device=str(device),
                  prior_L_kwargs=row, conditional_kwargs=graph.options,
                  resolved_training=graph.training[0], setup_seconds=setup_seconds,
                  sweep_seconds=times, total_fit_seconds=sum(times),
                  warmed_sweep_seconds=statistics.median(times[1:]) if len(times) > 1 else None,
                  signal_rmse=rmse)
    print(f"    signal RMSE={rmse:.5f}; total fit={sum(times):.2f}s", flush=True)
    return m, record


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Python environment has no available CUDA runtime. "
                           "Use --device cpu for a CPU test or run in your CUDA-enabled environment.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    metadata = dict(torch=torch.__version__, package=cebmf_torch.__file__, device=str(device),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    configuration=vars(args).copy(), results=[])
    print(json.dumps({k: v for k, v in metadata.items() if k != "results"}, indent=2), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in range(1, args.seeds + 1):
        # Generate once on CPU to preserve the notebook RNG, upload once per seed.
        y, truth, _, _ = simulate(seed, args.scenario, n=args.n, p=args.p)
        y, truth = y.to(device), truth.to(device)
        print(f"Seed {seed}: {args.n} rows, {args.p} columns, rank {args.rank}", flush=True)
        cold, cold_record = fit_stage(y, truth, args.cold_prior, args.stage_steps[0], args, seed)
        # Keep only the small loading matrix for optional plotting/reporting.
        cold_l = cold.L.detach().clone()
        del cold
        precursor, precursor_record = fit_stage(y, truth, "cgb", args.stage_steps[1], args, seed)
        warm, warm_record = fit_stage(y, truth, "cgb_sharp_2", args.stage_steps[2], args, seed, precursor)
        del precursor
        metadata['results'].append(dict(seed=seed, cold=cold_record, precursor=precursor_record, warm=warm_record))
        (output / 'timings.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        torch.save(dict(cold_L=cold_l, warm_L=warm.L.detach()), output / f'loadings_seed{seed}.pt')
        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # Explicit host conversion at the plotting boundary, outside timings.
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for ax, loading, title in zip(axes, (cold_l, warm.L),
                                         (f'{args.cold_prior}: cold', 'cgb -> cgb_sharp_2')):
                ax.imshow(loading.detach().cpu().numpy(), aspect='auto', interpolation='nearest', cmap='viridis')
                ax.set_title(title)
                ax.set_xlabel('Loading column')
            axes[0].set_ylabel('Row')
            fig.savefig(output / f'loadings_seed{seed}.png', dpi=140)
            plt.close(fig)
        del warm, cold_l


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--preset', choices=['compact', 'original'], default='compact')
    p.add_argument('--approximation', choices=['quadratic', 'quadrature'], default='quadratic')
    p.add_argument('--seeds', type=int, default=1, help='Number of simulation replicates, starting at seed 1.')
    p.add_argument('--stage-steps', type=int, nargs=3, default=[3, 2, 3], metavar=('COLD', 'CGB', 'WARM'))
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--p', type=int, default=200)
    p.add_argument('--rank', type=int, default=6)
    p.add_argument('--scenario', choices=['notebook', 'corrected'], default='notebook')
    p.add_argument('--cold-prior', choices=['spiked_emdn', 'cgb_sharp_2'], default='spiked_emdn')
    p.add_argument('--plot', action='store_true', help='Save heatmaps after fitting; no blocking plot windows.')
    p.add_argument('--output', default='output/tree_cuda_example')
    args = p.parse_args()
    if args.seeds < 1 or min(args.stage_steps) < 1 or args.rank < 2 or args.n < 4 or args.n % 4 or args.p < 2:
        p.error('Use positive seeds/stage steps, rank >= 2, n >= 4 divisible by 4, and p >= 2.')
    run(args)
