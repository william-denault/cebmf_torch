"""Training controls and the common spike-regularized neural-prior loss."""

import math

import torch


TRAINING_KEYS = {"penalty", "n_epochs", "batch_size", "lr"}


def training_options(name, raw, overrides, defaults):
    supplied = {**raw, **(overrides or {})}
    result = {
        "penalty": supplied.get("penalty", 1.0),
        "n_epochs": supplied.get("n_epochs", defaults["steps"]),
        "batch_size": supplied.get("batch_size"),
        "lr": supplied.get("lr", defaults["lr"]),
        "pretrain_epochs": supplied.get("n_epochs", defaults["pretrain_steps"]),
        "pretrain_lr": supplied.get("lr", 0.01),
    }
    penalty = result["penalty"]
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 1:
        raise ValueError("Joint learned-prior penalty must be finite and >= 1 (1 disables spike regularization).")
    if name == "emdn" and penalty != 1:
        raise ValueError("emdn has no spike to penalize; use penalty=1 or spiked_emdn.")
    epochs, batch = result["n_epochs"], result["batch_size"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("prior n_epochs must be a nonnegative integer.")
    if batch is not None and (isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        raise ValueError("prior batch_size must be a positive integer or None for a full batch.")
    if not isinstance(result["lr"], (int, float)) or not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("prior lr must be finite and positive.")
    return result


def spike_log_factor(mixture, penalty):
    """Log auxiliary likelihood pi_0(parents) ** (penalty - 1).

    For penalty >= 1 this is the probability of an observed auxiliary event.
    Including it in both latent transitions and parameter learning gives a
    common regularized target. It is constant in this node's own value, but
    varies when a latent parent is changed.
    """
    if penalty == 1:
        return mixture.log_weight.new_zeros(len(mixture.mean))
    return (penalty - 1) * mixture.log_weight[:, 0]


def fit_mixture_network(prior, inputs, values, *, components=None, precision=None,
                        n_epochs, batch_size, lr, penalty, generator):
    """Optimize complete latent-data loss, or normal-means initialization loss.

    One epoch visits every saved (draw, sample) pair once. The unpenalized
    loss and regularizer are both per-pair averages, independent of batch size.
    Keep the best *full-dataset* state after an epoch, including the start.
    """
    size = len(values)
    batch_size = size if batch_size is None else min(batch_size, size)

    def losses(index):
        mixture = prior(inputs[index])
        if components is None:
            a = torch.full_like(values[index], precision)
            _, score = mixture.posterior(a, a * values[index])
        else:
            score = mixture.log_prob(values[index], components[index])
        return -score - spike_log_factor(mixture, penalty)

    @torch.no_grad()
    def objective():
        total = values.new_zeros(())
        for start in range(0, size, batch_size):
            total += losses(slice(start, start + batch_size)).sum()
        return float(total / size)

    initial = objective()
    if not math.isfinite(initial):
        raise FloatingPointError("Nonfinite initial neural-prior loss.")
    best = initial
    best_state = {key: value.detach().clone() for key, value in prior.state_dict().items()}
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    updates, completed_epochs = 0, 0
    for _ in range(n_epochs):
        # Preserve the old full-batch schedule without consuming shuffle RNG.
        order = torch.randperm(size, generator=generator) if batch_size < size else torch.arange(size)
        failed = False
        try:
            for start in range(0, size, batch_size):
                optimizer.zero_grad()
                loss = losses(order[start:start + batch_size]).mean()
                if not torch.isfinite(loss):
                    failed = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
                optimizer.step()
                updates += 1
            updated = objective() if not failed else math.inf
        except FloatingPointError:
            break
        if not math.isfinite(updated):
            break
        completed_epochs += 1
        if updated < best:
            best = updated
            best_state = {key: value.detach().clone() for key, value in prior.state_dict().items()}
    prior.load_state_dict(best_state)
    return {"before": initial, "after": best, "n_epochs": n_epochs,
            "completed_epochs": completed_epochs, "batch_size": batch_size,
            "optimizer_steps": updates, "penalty": penalty}
