"""Closed-form cubic-smoothing-spline variance.

Two claims, each checked independently:

1. `iwp2_kernel` (the m=2 derivative-penalty reproducing kernel, a.k.a. the
   covariance of twice-integrated Brownian motion) plugged into
   `saddle_point_solve` with `nugget=lambda` directly -- no kappa, no
   `4*kappa**3` rescaling -- reproduces `scipy.interpolate.
   make_smoothing_spline(lam=lambda)`'s mean, to numerical-solve precision.
2. `saddle_point_variance` with that same kernel gives a predictive variance
   that is exactly the `kappa->0` limit of the *rescaled* Matern-3/2
   Universal-Kriging variance -- rescaled by the same `4*kappa**3` factor
   Corollary 5's nugget formula already uses. This rescaling is necessary
   because, unlike the mean, variance is not invariant to the kernel's
   absolute scale: the raw Matern-3/2 UK variance (sigma^2=1 convention)
   vanishes as kappa**3, not to a finite limit, until divided back out.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline

from smoothing_splines_uq.kernels import iwp2_kernel, matern32
from smoothing_splines_uq.uk import (
    poly_basis,
    saddle_point_solve,
    saddle_point_predict,
    saddle_point_variance,
)


def _setup():
    rng = np.random.default_rng(0)
    n = 40
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * x) + 0.1 * rng.standard_normal(n)
    x_dense = np.linspace(x.min(), x.max(), 100)
    return x, y, x_dense


def test_iwp2_kernel_reproduces_scipy_smoothing_spline_mean():
    x, y, x_dense = _setup()
    lam = 1e-2

    target = make_smoothing_spline(x, y, lam=lam)(x_dense)

    K = iwp2_kernel(x[:, None], x[None, :])
    Kt = iwp2_kernel(x_dense[:, None], x[None, :])
    P = poly_basis(x, degree=1)
    Pt = poly_basis(x_dense, degree=1)

    a, beta = saddle_point_solve(K, P, y, nugget=lam)
    f = saddle_point_predict(Kt, Pt, a, beta)

    max_err = np.max(np.abs(f - target))
    assert max_err < 1e-9, f"max_err={max_err:.3e} (data range {np.ptp(y):.3f})"


def test_rescaled_matern_variance_converges_to_iwp2_variance():
    x, y, x_dense = _setup()
    lam = 1e-2

    K = iwp2_kernel(x[:, None], x[None, :])
    Kt = iwp2_kernel(x_dense[:, None], x[None, :])
    Kss = iwp2_kernel(x_dense, x_dense)
    P = poly_basis(x, degree=1)
    Pt = poly_basis(x_dense, degree=1)
    var_G = saddle_point_variance(K, P, lam, Kt, Pt, Kss)

    kappas = [0.3, 0.1, 0.03, 0.01]
    errs = []
    for kappa in kappas:
        Km = matern32(np.abs(x[:, None] - x[None, :]), kappa)
        Kmt = matern32(np.abs(x_dense[:, None] - x[None, :]), kappa)
        Kmss = np.ones(len(x_dense))  # matern32(0) == 1
        nugget_m = 4 * kappa**3 * lam

        var_m_raw = saddle_point_variance(Km, P, nugget_m, Kmt, Pt, Kmss)
        var_m_rescaled = var_m_raw / (4 * kappa**3)
        errs.append(np.max(np.abs(var_m_rescaled - var_G)))

    errs = np.array(errs)
    rng_G = np.ptp(var_G)
    # monotone decay as kappa -> 0 (same style of check as Phase A3's slope
    # test), and tight by the smallest kappa tried.
    assert np.all(np.diff(errs) < 0), f"errs not monotone decaying: {errs}"
    assert errs[-1] / rng_G < 1e-2, f"relative error {errs[-1] / rng_G:.3e} too large"
