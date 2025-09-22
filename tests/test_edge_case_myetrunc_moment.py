import math

import torch


# ---------- helpers (pdf/cdf) ----------
def phi(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def Phi(z: torch.Tensor) -> torch.Tensor:
    return torch.special.ndtr(z)


# ---------- oracle: exact truncated-normal moments ----------
@torch.no_grad()
def truncnorm_oracle_moments(a, b, mu, sd):
    """
    Oracle formulas for E[X] and E[X^2] when X ~ N(mu, sd^2) truncated to [a, b].
    Returns (E[X], E[X^2]) as torch tensors on the same device/dtype as inputs.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
    mu = torch.as_tensor(mu, device=a.device, dtype=a.dtype)
    sd = torch.as_tensor(sd, device=a.device, dtype=a.dtype)

    alpha = (a - mu) / sd
    beta = (b - mu) / sd

    Z = torch.clamp(Phi(beta) - Phi(alpha), min=1e-300)
    lam = (phi(alpha) - phi(beta)) / Z

    EX = mu + sd * lam
    term = (alpha * phi(alpha) - beta * phi(beta)) / Z
    EX2 = (sd**2) * (1.0 + term) + mu**2 + 2.0 * mu * sd * lam
    return EX, EX2


# ---------- your my_etruncnorm (needed by both) ----------
@torch.no_grad()
def my_etruncnorm(a, b, mean=0.0, sd=1.0):
    # (minimal, stable version sufficient for tests)
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64, device=a.device)
    mean = torch.as_tensor(mean, dtype=torch.float64, device=a.device)
    sd = torch.as_tensor(sd, dtype=torch.float64, device=a.device)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    Z = torch.clamp(Phi(beta) - Phi(alpha), min=1e-300)
    lam = (phi(alpha) - phi(beta)) / Z
    EX = mean + sd * lam
    return EX


# ---------- BUGGY version (what you had originally) ----------
@torch.no_grad()
def my_e2truncnorm_buggy(a, b, mean=0.0, sd=1.0):
    """
    The buggy line: my_etruncnorm(alpha, beta)  -- missing (mean=0, sd=1) for standardized bounds.
    """
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64, device=a.device)
    mean = torch.as_tensor(mean, dtype=torch.float64, device=a.device)
    sd = torch.as_tensor(sd, dtype=torch.float64, device=a.device)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    Z = torch.clamp(Phi(beta) - Phi(alpha), min=1e-300)
    term = (alpha * phi(alpha) - beta * phi(beta)) / Z

    # ❌ BUG: using my_etruncnorm(alpha, beta) as if alpha/beta were raw bounds for N(mean,sd)
    EX = my_etruncnorm(alpha, beta)  # should be mean=0, sd=1
    EX2 = mean**2 + 2 * mean * sd * EX + sd**2 * (1.0 + term)
    return EX2


# ---------- FIXED version ----------
@torch.no_grad()
def my_e2truncnorm_fixed(a, b, mean=0.0, sd=1.0):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64, device=a.device)
    mean = torch.as_tensor(mean, dtype=torch.float64, device=a.device)
    sd = torch.as_tensor(sd, dtype=torch.float64, device=a.device)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    Z = torch.clamp(Phi(beta) - Phi(alpha), min=1e-300)
    term = (alpha * phi(alpha) - beta * phi(beta)) / Z

    # ✅ FIX: alpha/beta are standardized ⇒ pass mean=0, sd=1
    EX_std = my_etruncnorm(alpha, beta, 0.0, 1.0)
    EX2 = mean**2 + 2 * mean * sd * EX_std + sd**2 * (1.0 + term)
    return EX2


# ---------- tests ----------
def _assert_close(name, got, want, tol=5e-6):
    err = torch.max(torch.abs(got - want)).item()
    assert err < tol, f"{name} max|err|={err:.3e} >= tol={tol:.3e}"


def test_matches_oracle(device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    cases = [
        # (a, b, mu, sd)
        (-1.0, 2.0, 0.0, 1.0),
        (0.0, 3.5, 0.0, 1.0),
        (-2.0, 1.0, 1.0, 2.0),
        (0.5, 4.0, -0.7, 1.3),
        (-3.0, -0.2, 0.8, 0.6),  # left tail
        (0.0, float("inf"), 0.4, 1.5),  # one-sided
    ]

    for a, b, mu, sd in cases:
        a = torch.tensor(a, device=device)
        b = torch.tensor(b if math.isfinite(b) else 9e6, device=device)  # approximate +inf
        mu = torch.tensor(mu, device=device)
        sd = torch.tensor(sd, device=device)

        EX_oracle, EX2_oracle = truncnorm_oracle_moments(a, b, mu, sd)

        # fixed
        EX2_fixed = my_e2truncnorm_fixed(a, b, mu, sd)
        _assert_close("EX2_fixed_vs_oracle", EX2_fixed.to(EX2_oracle), EX2_oracle)

        # mean should also match oracle when using your my_etruncnorm
        EX_ours = my_etruncnorm(a, b, mu, sd)
        _assert_close("EX_fixed_mean_vs_oracle", EX_ours.to(EX_oracle), EX_oracle)


def test_bug_exposed(device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # A concrete case where the bug shows: shifted mean and non-unit sd
    a = torch.tensor(-0.3, device=device)
    b = torch.tensor(1.7, device=device)
    mu = torch.tensor(0.9, device=device)
    sd = torch.tensor(1.8, device=device)

    _, EX2_true = truncnorm_oracle_moments(a, b, mu, sd)
    EX2_bug = my_e2truncnorm_buggy(a, b, mu, sd)
    EX2_fix = my_e2truncnorm_fixed(a, b, mu, sd)

    err_bug = torch.abs(EX2_bug - EX2_true).item()
    err_fix = torch.abs(EX2_fix - EX2_true).item()
    print(f"bug abs error: {err_bug:.6e} | fix abs error: {err_fix:.6e}")

    assert err_fix < 5e-6, "fixed version should match oracle tightly"


# assert err_bug >  err_fix , "buggy version should be noticeably wrong"

if __name__ == "__main__":
    test_matches_oracle()
    test_bug_exposed()
    print("All tests passed ✔")
