"""Summarize saved tree benchmark replicates without refitting any models."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_tree_priors import simulate, rmse

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/tree_prior_benchmark"
METHODS = ["gbinary", "cgb", "cgb_plugin", "cgb_self", "spiked", "spiked_self"]
LABELS = {"gbinary": "GB (no covariates)", "cgb": "CGB (no covariates)",
          "cgb_plugin": "CGB plug-in", "cgb_self": "CGB joint VI",
          "spiked": "Spiked EMDN (no covariates)", "spiked_self": "Spiked EMDN joint VI"}
COLORS = dict(zip(METHODS, ["#4b5563", "#b18a41", "#ab6576", "#217b9a", "#9c72b0", "#26835b"]))


def load_records(directory=OUTPUT / "main"):
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("complete"):
            record["path"] = str(path)
            records.append(record)
    return records


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["config"]["scenario"], record["config"]["method"])].append(record)
    summaries = []
    for scenario in ("notebook", "corrected"):
        for method in METHODS:
            group = groups[scenario, method]
            if not group:
                continue
            rmse = [r["history"][-1]["signal_rmse"] for r in group]
            seconds = [r["history"][-1]["seconds"] for r in group]
            summaries.append(dict(scenario=scenario, method=method, n=len(group),
                                  rmse_mean=statistics.mean(rmse),
                                  rmse_sd=statistics.stdev(rmse) if len(rmse) > 1 else 0,
                                  seconds_median=statistics.median(seconds),
                                  sigma_mean=statistics.mean(r["history"][-1]["sigma"] for r in group)))
    return groups, summaries


def make_figures(groups, summaries):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, scenario in zip(axes, ("notebook", "corrected")):
        for i, method in enumerate(METHODS):
            group = groups[scenario, method]
            if not group:
                continue
            values = [r["history"][-1]["signal_rmse"] for r in group]
            ax.scatter(np.arange(len(values)) * .035 + i - .07, values,
                       color=COLORS[method], alpha=.7, s=30, zorder=3)
            ax.plot([i-.22, i+.22], [np.mean(values)]*2, color=COLORS[method], lw=3)
        ax.set_xticks(range(len(METHODS)), [LABELS[m].replace(" ", "\n") for m in METHODS])
        ax.set_ylabel("Noiseless-signal RMSE (lower is better)")
        ax.set_title("Original notebook simulation" if scenario == "notebook" else "Corrected four-leaf tree")
        ax.grid(axis="y", alpha=.18)
        ax.set_ylim(bottom=0)
    fig.suptitle("30 sweeps, matched SVD factors, ash columns; dots are independent simulation seeds")
    fig.savefig(OUTPUT / "rmse_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, scenario in enumerate(("notebook", "corrected")):
        for method in METHODS:
            group = groups[scenario, method]
            if not group:
                continue
            values = np.array([[h["signal_rmse"] for h in r["history"]] for r in group])
            seconds = np.array([[h["seconds"] for h in r["history"]] for r in group])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            for col, x in enumerate((np.arange(values.shape[1]), np.median(seconds, axis=0))):
                axes[row, col].plot(x[1:], mean[1:], label=LABELS[method], color=COLORS[method])
                axes[row, col].fill_between(x[1:], (mean-sd)[1:], (mean+sd)[1:],
                                           color=COLORS[method], alpha=.1)
        for col in range(2):
            axes[row, col].set_title("Notebook" if scenario == "notebook" else "Corrected tree")
            axes[row, col].set_ylabel("Signal RMSE; mean and SD across seeds")
            axes[row, col].set_xlabel("Variational sweep" if col == 0 else "Elapsed seconds (concurrent CPU runs)")
            axes[row, col].set_ylim(0, .27)
            axes[row, col].grid(alpha=.15)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("GB's initial errors exceed the displayed vertical range")
    fig.savefig(OUTPUT / "learning_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(METHODS)+1, figsize=(14, 7), constrained_layout=True)
    for row, scenario in enumerate(("notebook", "corrected")):
        for col, method in enumerate(METHODS):
            records = [r for r in groups[scenario, method] if r["config"]["seed"] == 1]
            if not records:
                continue
            result = torch.load(Path(records[0]["path"]).with_suffix(".pt"), weights_only=True)
            if col == 0:
                axes[row, 0].imshow(result["truth_L"].numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
                axes[row, 0].set_title("Simulated L (7 columns)")
            loading = result["L"].numpy()
            # Display each fitted loading on its own RMS scale; never score against
            # named true factors because the redundant tree is not identifiable.
            loading = loading / np.maximum(np.sqrt(np.mean(loading**2, axis=0)), 1e-8)
            fitted_image = axes[row, col+1].imshow(loading, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-2, vmax=2)
            axes[row, col+1].set_title(LABELS[method], fontsize=9)
        axes[row, 0].set_ylabel(("Notebook" if row == 0 else "Corrected") + "\nCell index")
        for ax in axes[row]:
            ax.set_xlabel("Loading column")
        axes[row, 0].set_xticks(range(7), range(1, 8))
        for ax in axes[row, 1:]:
            ax.set_xticks(range(6), range(1, 7))
    fig.colorbar(fitted_image, ax=axes[:, 1:].ravel().tolist(), shrink=.7, pad=.01,
                 label="Fitted loading / column RMS (clipped at +/-2)")
    fig.suptitle("Seed 1: fitted columns have arbitrary order/scale and need not equal the seven simulated programs")
    fig.savefig(OUTPUT / "loadings_seed1.png", dpi=170)
    plt.close(fig)


def main():
    records = load_records()
    groups, summaries = summarize(records)
    if len(records) != 60:
        raise RuntimeError(f"Expected 60 completed main fits; found {len(records)}")
    for group in groups.values():
        assert sorted(r["config"]["seed"] for r in group) == [1, 2, 3, 4, 5]
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    paired = []
    for scenario in ("notebook", "corrected"):
        for first, second in (("cgb_self", "cgb_plugin"), ("spiked_self", "gbinary"),
                              ("spiked_self", "spiked")):
            a = {r["config"]["seed"]: r["history"][-1]["signal_rmse"] for r in groups[scenario, first]}
            b = {r["config"]["seed"]: r["history"][-1]["signal_rmse"] for r in groups[scenario, second]}
            differences = [a[seed] - b[seed] for seed in sorted(a)]
            paired.append(dict(scenario=scenario, first=first, second=second,
                               mean_difference=statistics.mean(differences),
                               sd_difference=statistics.stdev(differences),
                               wins=sum(d < 0 for d in differences), differences=differences))
    (OUTPUT / "paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    rare_rows = []
    for method in METHODS:
        for record in groups["notebook", method]:
            saved = torch.load(Path(record["path"]).with_suffix(".pt"), weights_only=True)
            _, signal, loading, _ = simulate(record["config"]["seed"], "notebook")
            prediction = saved["L"] @ saved["F"].T
            rare = loading[:, 3:].sum(1) == 0
            rare_rows.append(dict(method=method, seed=record["config"]["seed"],
                                  common_rmse=rmse(prediction[~rare], signal[~rare]),
                                  exceptional_rmse=rmse(prediction[rare], signal[rare])))
    (OUTPUT / "exceptional_rows.json").write_text(json.dumps(rare_rows, indent=2), encoding="utf-8")
    lines = ["| Simulation | Row prior / update | Signal RMSE, mean +/- SD | Median seconds |",
             "| --- | --- | ---: | ---: |"]
    for r in summaries:
        lines.append(f"| {r['scenario']} | {LABELS[r['method']]} | {r['rmse_mean']:.4f} +/- {r['rmse_sd']:.4f} | {r['seconds_median']:.1f} |")
    (OUTPUT / "RESULTS_TABLE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    make_figures(groups, summaries)


if __name__ == "__main__":
    main()
