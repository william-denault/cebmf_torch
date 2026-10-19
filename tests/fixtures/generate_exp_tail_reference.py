"""Generate independent 80-digit quadrature references; not run during pytest.

Integrates exp(-rate * theta) * N(x | theta, s²) on theta >= 0.
Rescaling the integration variable keeps its mass away from a vanishingly
small interval. No normal CDF, Mills ratio, or continued fraction is used.
Requires mpmath, already a transitive dependency of PyTorch via SymPy.
"""

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 80


def reference(x, s, rate):
    x, s, rate = map(mp.mpf, map(str, (x, s, rate)))
    z, t = x / s, s * rate - x / s
    if t >= 0:
        h = s / max(1, t)

        def kernel(u):
            return mp.exp(-t * h / s * u - (h / s * u) ** 2 / 2)

        intervals = [0, 1, 4, 16, mp.inf]
        log_prefactor = mp.log(rate * h / s) - z**2 / 2 - mp.log(2 * mp.pi) / 2
    else:
        h = s

        def kernel(u):
            return mp.exp(-((u + t) ** 2) / 2)

        intervals = sorted({mp.mpf(0), max(mp.mpf(0), -t - 8), -t, -t + 8, mp.inf})
        log_prefactor = mp.log(rate) + (t**2 - z**2) / 2 - mp.log(2 * mp.pi) / 2
    integrals = [mp.quad(lambda u, power=k: u**power * kernel(u), intervals) for k in range(3)]
    mean = h * integrals[1] / integrals[0]
    mean2 = h**2 * integrals[2] / integrals[0]
    return {
        "x": float(x),
        "s": float(s),
        "rate": float(rate),
        "log_density": float(log_prefactor + mp.log(integrals[0])),
        "mean": float(mean),
        "mean2": float(mean2),
        "gradient_x": float((mean - x) / s**2),
        "gradient_s": float((mean2 - 2 * x * mean + x**2 - s**2) / s**3),
        "gradient_rate": float(1 / rate - mean),
    }


cases = [(0, 500000, rate) for rate in (0.01, 1, 100)]
cases += [
    (x, s, rate)
    for x, s, rate in (
        (-20, 0.1, 1),
        (20, 0.1, 1),
        (0, 1, 1),
        (3, 2, 0.5),
        (1, 500000, 1),
        (-1, 500000, 1),
        (500000, 500000, 1),
        (-500000, 500000, 1),
        (0, 1e12, 2),
        (10, 2, 5),
        (-10, 2, 0.5),
    )
]
# Both the regular/tail switch and the near-noise log1p switch.
cases += [(1 - t, 1, 1) for t in (3.9999, 4, 4.0001, 8, 50)]
cases += [(z, 1, 12) for z in (5.9999, 6, 6.0001)]
output = Path(__file__).with_name("exp_tail_reference.json")
output.write_text(json.dumps([reference(*case) for case in cases], indent=2) + "\n", encoding="utf-8")
