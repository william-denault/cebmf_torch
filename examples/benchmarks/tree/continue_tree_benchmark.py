"""Optimizer-effort diagnostic from a saved conditional benchmark checkpoint."""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import torch

from benchmark_tree_priors import simulate, metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.checkpoint).replace(".model.pt", ".json")).read_text())
    config = metadata["config"]
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    observed, signal, _, _ = simulate(config["seed"], config["scenario"], config["n"], config["p"])
    test = ~model.mask.bool()
    for training in model.conditional_fit.row.training:
        training["n_epochs"] = args.epochs
    history = [metrics(model, signal, observed, test, 0, 0)]
    output = dict(checkpoint=str(args.checkpoint), epochs=args.epochs,
                  steps=args.steps, complete=False, history=history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        with contextlib.redirect_stdout(io.StringIO()):
            model.iter_once()
        history.append(metrics(model, signal, observed, test, time.perf_counter()-start, step))
        args.output.write_text(json.dumps(output, indent=2))
        if step == 1 or step % 5 == 0:
            print(step, history[-1], flush=True)
    output["complete"] = True
    args.output.write_text(json.dumps(output, indent=2))
