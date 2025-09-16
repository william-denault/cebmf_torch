"""Mathematical utilities and helper functions."""

# Submodules organized by functionality
from . import device, distribution_operation, maths, mixture, posterior

# Most commonly used functions from each module
from .device import get_device
from .maths import my_e2truncnorm, my_etruncnorm, safe_tensor_to_float
from .mixture import autoselect_scales_mix_norm, optimize_pi_logL
from .posterior import posterior_mean_exp, posterior_mean_norm

__all__ = [
    # Submodules
    "device",
    "distribution_operation",
    "maths",
    "mixture",
    "posterior",
    # Common functions
    "get_device",
    "my_e2truncnorm",
    "my_etruncnorm",
    "safe_tensor_to_float",
    "autoselect_scales_mix_norm",
    "optimize_pi_logL",
    "posterior_mean_exp",
    "posterior_mean_norm",
]