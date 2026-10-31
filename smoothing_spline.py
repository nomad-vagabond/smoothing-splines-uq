"""Wrapper classes giving scipy's smoothing splines a predictive variance.

Don't patch scipy, wrap it: each class below calls the matching scipy constructor
unchanged for the mean (so it is bit-identical to scipy's own answer -- see
the docstrings for the exact numbers), and separately solves Theorem 2a's
bordered system with the matching closed-form reproducing kernel
(`iwp2_kernel` / `tps2d_kernel`) for the variance scipy does not expose.

Both classes expose the same two-method API: `.mean(xq)` and `.variance(xq)`.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline, RBFInterpolator

from .kernels import iwp2_kernel, tps2d_kernel, poly_basis_2d
from .uk import poly_basis, saddle_point_solve, saddle_point_variance


class SmoothingSpline1D:
    """`scipy.interpolate.make_smoothing_spline`, with a companion variance.

    `.mean(xq)` calls scipy's own fitted `BSpline` directly -- identical to
    `make_smoothing_spline(x, y, lam=lam)(xq)` to the solver's own precision.
    `.variance(xq)` solves Theorem 2a's bordered system with `iwp2_kernel`
    (the m=2 cubic-penalty reproducing kernel, Kimeldorf & Wahba 1970) and
    `nugget=lam` directly -- no kappa, no rescaling. See
    `notebooks/spline_variance_without_kriging.ipynb` for the numerical
    confirmation that this variance is the kappa->0 limit of Matern-3/2
    Universal-Kriging's own variance (both via `krig_splines`'s own solver
    and independently via SMT's `KRG`).

    Only the m=2 (cubic) case is implemented, matching `iwp2_kernel`.
    """

    def __init__(self, x, y, lam):
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.lam = float(lam)

        self._spline = make_smoothing_spline(self.x, self.y, lam=self.lam)
        self._K = iwp2_kernel(self.x[:, None], self.x[None, :])
        self._P = poly_basis(self.x, degree=1)

    def mean(self, xq):
        return self._spline(np.asarray(xq, dtype=float))

    def variance(self, xq):
        xq = np.asarray(xq, dtype=float).reshape(-1)
        Kq = iwp2_kernel(xq[:, None], self.x[None, :])
        Pq = poly_basis(xq, degree=1)
        Kqq = iwp2_kernel(xq, xq)  # k(x*,x*) per query point
        return saddle_point_variance(self._K, self._P, self.lam, Kq, Pq, Kqq)


class SmoothingSpline2D:
    """`scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline")`, with
    a companion variance.

    `.mean(Xq)` calls scipy's own fitted `RBFInterpolator` directly --
    identical to `RBFInterpolator(X, y, kernel="thin_plate_spline",
    smoothing=lam, degree=1)(Xq)` to the solver's own precision (`~1e-14` in
    `notebooks/spline_variance_2d_vs_smt.ipynb`). `.variance(Xq)` solves the
    *same* bordered system scipy's own `RBFInterpolator` docstring writes out
    -- `(K+lam*I)a + P*b = y`, `P^T a = 0` -- with `tps2d_kernel` (Duchon
    1977's 2-D bending-energy reproducing kernel) and `nugget=lam` directly.

    Only `degree=1` (the minimum scipy allows for this kernel, and the
    degree used throughout this repo) is implemented.
    """

    def __init__(self, X, y, lam):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.lam = float(lam)

        self._rbf = RBFInterpolator(self.X, self.y, kernel="thin_plate_spline",
                                     smoothing=self.lam, degree=1)
        self._K = tps2d_kernel(self.X, self.X)
        self._P = poly_basis_2d(self.X)

    def mean(self, Xq):
        return self._rbf(np.asarray(Xq, dtype=float))

    def variance(self, Xq):
        Xq = np.asarray(Xq, dtype=float)
        Kq = tps2d_kernel(Xq, self.X)
        Pq = poly_basis_2d(Xq)
        Kqq = np.zeros(len(Xq))  # tps2d_kernel(x,x) = 0 by convention
        return saddle_point_variance(self._K, self._P, self.lam, Kq, Pq, Kqq)
