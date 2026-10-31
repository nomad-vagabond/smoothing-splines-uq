"""N-D `csaps.NdGridCubicSmoothingSpline` with Bayesian posterior variance.

Merges what were two independently-developed, independently-validated
implementations in the exploratory (`krig_splines`) codebase --
`csaps_grid_uq.py` and `grid_smooth_spline_uq.py` -- into a single class with
two selectable off-grid evaluation strategies, rather than two classes users
had to choose between by trial. Both strategies compute the *same* exact,
closed-form posterior mean/variance ON the training grid
($\\hat Y = Z \\times_1 S_1 \\times_2 S_2 \\cdots \\times_N S_N$,
$\\mathrm{Var}(F[p_1,\\ldots,p_N])=\\sigma^2 \\prod_i S_i[p_i,p_i]$, the direct
N-way generalization of the 2-D formulas derived in
`05_csaps_variance_derivation.ipynb` and confirmed by brute-force Kronecker
construction for N=2 (`02_csaps_mean_and_variance.ipynb` Test 4) and spot-checked
the same way for N=3, now locked in by `tests/test_csaps_uq.py`); they differ
only in how *off-grid* (scattered) query points in `.predict()` are handled:

- `off_grid_method="exact"` (default, any number of dimensions): evaluate
  the per-axis `PolyharmonicSmoothingSpline(m=2, d=1)` kernel exactly on
  each axis's *unique* query coordinates, gathered back to the requested
  points. No approximation, at the cost of `O(n_unique)` work per axis per
  call.
- `off_grid_method="fast"` (2-D only): build a
  `scipy.interpolate.RectBivariateSpline` re-interpolant of the mean/variance
  surfaces once, at construction time, and evaluate off-grid queries through
  it (a compiled FITPACK routine). Confirmed in this project's own speed
  benchmark (`10_variance_prediction_speed_benchmark.ipynb`) to be
  `~55x-63x` faster than `"exact"` off-grid, at the cost of a small, regime-
  dependent re-interpolation error off-grid (`~1e-5` for a smooth surface up
  to `~1e-1` near interpolation, `06_csaps_variance_implementation_comparison.ipynb`,
  `02_csaps_mean_and_variance.ipynb` Test 6). Exact ON the training grid
  either way -- this trade-off only applies to scattered queries that don't
  coincide with the grid. `RectBivariateSpline` is inherently a 2-D
  re-interpolant (there is no direct N-D FITPACK equivalent this project
  has adopted) -- `"fast"` is refused outright for any other dimension count
  rather than silently falling back to `"exact"` or guessing at an N-D
  re-interpolation scheme never validated here.
"""

from functools import reduce

import numpy as np
from csaps import NdGridCubicSmoothingSpline
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from .polyharmonic import PolyharmonicSmoothingSpline


class NdGridCubicSmoothingSplineUQ(NdGridCubicSmoothingSpline):
    """N-D `csaps.NdGridCubicSmoothingSpline` with Bayesian posterior variance.

    Parameters
    ----------
    xdata : [x1, x2, ..., xN]
        1-D per-axis grid coordinate arrays.
    ydata : (len(x1), len(x2), ..., len(xN)) array
        Grid response values.
    smooth : float, tuple of N floats, or None
        `csaps`'s own smoothing parameter(s) $p\\in(0,1]$ per axis.
    sigma2 : "wahba" (default) or a known float
        Noise variance. `"wahba"` uses the GCV-style
        $\\mathrm{RSS}/(\\prod_i n_i-\\mathrm{edf})$ estimator,
        $\\mathrm{edf}=\\prod_i\\mathrm{tr}(S_i)$ (exact, since the smoother
        is an N-way Kronecker product).
    off_grid_method : "exact" (default, any N) or "fast" (2-D only)
        See module docstring.
    """

    def __init__(self, xdata, ydata, smooth=None, sigma2="wahba", off_grid_method="exact"):
        super().__init__(xdata, ydata, smooth=smooth)
        self.ndim = len(xdata)
        if off_grid_method not in ("exact", "fast"):
            raise ValueError(f"off_grid_method must be 'exact' or 'fast', got {off_grid_method!r}")
        if off_grid_method == "fast" and self.ndim != 2:
            raise NotImplementedError(
                f"off_grid_method='fast' is 2-D only (RectBivariateSpline has no N-D "
                f"equivalent adopted here) -- got {self.ndim} axes. Use 'exact', which "
                f"supports any number of dimensions.")
        self.off_grid_method = off_grid_method

        xs = [np.asarray(a, float) for a in xdata]
        if not all(0 < p <= 1 for p in self.smooth):
            raise ValueError("variance requires smooth in (0, 1] on every axis (p=0 is a "
                              "degenerate lam=inf line fit, not handled here)")
        lams = tuple((1 - p) / p for p in self.smooth)
        self._lam = lams

        # Variance does not depend on the data y -- only on x-locations and lam.
        self._axes = [PolyharmonicSmoothingSpline(x.reshape(-1, 1), np.zeros_like(x), m=2, lam=lam, sigma2=1.0)
                      for x, lam in zip(xs, lams)]

        var_grids = [axis.variance(x.reshape(-1, 1)) for axis, x in zip(self._axes, xs)]
        self._edf = float(np.prod([v.sum() for v in var_grids]))

        n_total = int(np.prod([len(x) for x in xs]))
        Zhat = np.asarray(self(xdata))
        rss = float(np.sum((np.asarray(ydata) - Zhat) ** 2))
        if sigma2 == "wahba":
            self._sigma2 = rss / max(n_total - self._edf, 1e-8)
        else:
            self._sigma2 = float(sigma2)

        if self.off_grid_method == "fast":
            x1, x2 = xs
            mean_grid = Zhat
            var_grid = self.variance(xdata)
            self._mean_spline = RectBivariateSpline(x1, x2, mean_grid, kx=3, ky=3)
            self._var_spline = RectBivariateSpline(x1, x2, var_grid, kx=3, ky=3)

    @property
    def sigma2(self):
        return self._sigma2

    @property
    def edf(self):
        return self._edf

    def variance(self, xq):
        """Posterior variance alone, with no mean computed alongside it.

        Accepts either a grid-tuple query (`[x1_query, ..., xN_query]`,
        cartesian product, same convention as `__call__`) or a scattered
        `(N, ndim)` array of query points (same convention as `predict()`/
        `predict_mean()`). Grid-tuple queries are always exact;
        `off_grid_method` only affects the scattered case, exactly as it
        does for the mean in `predict_mean()` -- the two are independent
        computations (the "exact" scattered path re-evaluates each axis's
        own closed-form kernel on its unique query coordinates and gathers
        back; "fast" evaluates a separate, mean-only vs. variance-only
        `RectBivariateSpline` built once at construction time) and neither
        one requires the other to be computed first.
        """
        if isinstance(xq, (tuple, list)) and not isinstance(xq, np.ndarray):
            xqs = [np.asarray(a, float) for a in xq]
            vs = [axis.variance(x.reshape(-1, 1)) for axis, x in zip(self._axes, xqs)]
            return self._sigma2 * reduce(np.multiply.outer, vs)

        xq_arr = np.asarray(xq, float)
        if self.off_grid_method == "fast":
            x1_q, x2_q = xq_arr[:, 0], xq_arr[:, 1]
            return self._var_spline.ev(x1_q, x2_q)
        uniques, invs = zip(*(np.unique(xq_arr[:, i], return_inverse=True) for i in range(self.ndim)))
        var_grid = self.variance(list(uniques))
        return var_grid[tuple(invs)]

    def predict_mean(self, x_pred):
        """Mean (surface value) alone, with no variance computed alongside
        it. Same dual query convention as `predict()`/`variance()`: a
        grid-tuple (cartesian product) or a scattered `(N, ndim)` array.
        Grid-tuple queries are always exact; `off_grid_method` only
        affects the scattered case. This is exactly the mean half of what
        `predict()` already computed internally -- extracted here so it
        can be timed or called on its own, independently of `variance()`.
        """
        if isinstance(x_pred, (tuple, list)) and not isinstance(x_pred, np.ndarray):
            xqs = [np.asarray(a, float) for a in x_pred]
            return np.asarray(self(xqs))

        x_pred = np.asarray(x_pred, float)
        if self.off_grid_method == "fast":
            x1_p, x2_p = x_pred[:, 0], x_pred[:, 1]
            return self._mean_spline.ev(x1_p, x2_p)
        uniques, invs = zip(*(np.unique(x_pred[:, i], return_inverse=True) for i in range(self.ndim)))
        mean_grid = np.asarray(self(list(uniques)))
        return mean_grid[tuple(invs)]

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Predict mean/uncertainty.

        Parameters
        ----------
        x_pred : (N, ndim) array-like of scattered points, or a tuple/list
            of `ndim` 1-D arrays queried as a cartesian grid (same
            convention as __call__/variance). Grid-tuple queries are always
            exact; `off_grid_method` only affects the scattered `(N, ndim)`
            case, and only `"exact"` is available for `ndim != 2`.
        return_std, return_ci : bool
        alpha : float
            Significance level for the 95%-style intervals (default 0.05).

        Returns
        -------
        np.ndarray (just the mean) if neither flag is set, else a dict with
        'mean', 'var', 'std', and -- if return_ci -- 'ci_lower'/'ci_upper'
        (uncertainty of the mean surface) and 'pi_lower'/'pi_upper'
        (uncertainty of a new noisy observation, i.e. variance + sigma^2).

        A thin convenience wrapper around `predict_mean()` and `variance()`,
        kept for backward compatibility and for callers who genuinely want
        both; if only one quantity is needed, call `predict_mean()` or
        `variance()` directly instead, to avoid paying for the other.
        """
        mean = self.predict_mean(x_pred)
        var = np.clip(self.variance(x_pred), 0.0, None)
        std = np.sqrt(var)

        if not (return_std or return_ci):
            return mean

        res = {"mean": mean, "var": var, "std": std}
        if return_ci:
            z = norm.ppf(1.0 - alpha / 2.0)
            res["ci_lower"] = mean - z * std
            res["ci_upper"] = mean + z * std
            res["pi_lower"] = mean - z * np.sqrt(var + self.sigma2)
            res["pi_upper"] = mean + z * np.sqrt(var + self.sigma2)
        return res
