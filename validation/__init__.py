"""Tools used only for cross-checking in validation notebooks -- not part of
the main `smoothing_splines_uq` modeling API.

Each class here earns its place by having caught something real during
development (a bug, a robustness gap) via independent re-implementation,
Research Plan Phase F4's discipline -- not because it's a better everyday
tool than what's in the main package.
"""

from .rmtb_uncertainty_wrapper import RMTBUncertaintyWrapper

__all__ = ["RMTBUncertaintyWrapper"]
