"""smoothing_splines_uq: uncertainty quantification for N-D smoothing splines.

A cleaned, consolidated rebuild of an earlier exploratory codebase
("Splines and Kriging as One Surrogate Model" / the Gaussian RMTS
extension), keeping only confirmed results.

Public API
----------
Foundational (Theorem 2a/2b bordered system, the Kimeldorf-Wahba identity
everything else builds on):
    `saddle_point_solve`, `saddle_point_predict`, `gls_predict`,
    `saddle_point_variance` (`uk.py`)

General N-D Duchon/polyharmonic smoothing spline, with calibrated variance
(the theoretical anchor -- its 1-D, m=2 case is *exactly* a natural cubic
smoothing spline; its flat (kappa->0) limit is *exactly* isotropic Kriging,
Corollary 5):
    `PolyharmonicSmoothingSpline` (`polyharmonic.py`)

Thin wrappers giving scipy's own smoothing splines a companion variance,
bit-identical to scipy's mean:
    `SmoothingSpline1D`, `SmoothingSpline2D` (`smoothing_spline.py`)

Tensor-product (`csaps`-based) and finite-basis (SMT `RMTB`-based) UQ, the
two N-D spline families this project actually targets:
    `NdGridCubicSmoothingSplineUQ` (`csaps_uq.py`)
    `RMTBUQ` (`rmtb_uq.py`)

Plotting helpers (2-D maps, two-band 1-D CI/PI slices) live in `plotting.py`
and are not re-exported here -- import them explicitly from that module.

`validation/` holds tools used only for cross-checking in validation
notebooks, not part of the main modeling API -- see its own docstring.
"""

from .uk import saddle_point_solve, saddle_point_predict, gls_predict, saddle_point_variance
from .polyharmonic import PolyharmonicSmoothingSpline
from .smoothing_spline import SmoothingSpline1D, SmoothingSpline2D
from .csaps_uq import NdGridCubicSmoothingSplineUQ
from .rmtb_uq import RMTBUQ

__all__ = [
    "saddle_point_solve", "saddle_point_predict", "gls_predict", "saddle_point_variance",
    "PolyharmonicSmoothingSpline",
    "SmoothingSpline1D", "SmoothingSpline2D",
    "NdGridCubicSmoothingSplineUQ",
    "RMTBUQ",
]
