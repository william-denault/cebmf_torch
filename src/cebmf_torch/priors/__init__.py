"""Prior registry and prior classes for EBMF."""

# Submodules for different prior types
from . import base, learned, mixture, point

# Main registry for prior lookups
from .registry import PRIOR_REGISTRY

__all__ = [
    # Main registry
    "PRIOR_REGISTRY",
    # Prior modules
    "base",
    "learned",
    "mixture",
    "point",
]
