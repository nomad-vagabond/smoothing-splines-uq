"""Phase A1 -- saddle-point solve reproduces plain interpolation/ridge.

See "Research Plan - From Fundamentals to Gaussian RMTS.md", Phase A, A1.
"""
import numpy as np

from smoothing_splines_uq.kernels import brownian
from smoothing_splines_uq.uk import poly_basis, saddle_point_solve, saddle_point_predict


def _fit(x, y, nugget):
    K = brownian(x[:, None], x[None, :])
    P = poly_basis(x, degree=0)  # trend = span{1}
    return saddle_point_solve(K, P, y, nugget)


def test_a1_interpolation_is_the_natural_piecewise_linear_spline():
    """At nugget=0 (lambda=0), Theorem 2(a) must reproduce the natural
    piecewise-linear interpolant: the Brownian kernel's RKHS norm is
    \\int h'^2, whose minimal-norm interpolant is the linear spline, which
    is exactly `np.interp`'s straight-line-between-knots curve."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 6))
    y = rng.normal(size=6)

    a, beta = _fit(x, y, nugget=0.0)

    xt = np.linspace(x.min(), x.max(), 401)
    Kt = brownian(xt[:, None], x[None, :])
    Pt = poly_basis(xt, degree=0)
    f = saddle_point_predict(Kt, Pt, a, beta)

    f_ref = np.interp(xt, x, y)

    max_err = np.max(np.abs(f - f_ref))
    assert max_err <= 1e-8, f"max_err={max_err:.3e}"


def test_a1_large_nugget_limit_is_the_pure_trend():
    """As nugget -> inf, the kernel term is overwhelmed by the noise term
    and the fit reduces to the pure trend: for P=span{1}, that is the
    sample mean, everywhere."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 6))
    y = rng.normal(size=6)

    a, beta = _fit(x, y, nugget=1e8)

    xt = np.linspace(0, 1, 50)
    Kt = brownian(xt[:, None], x[None, :])
    Pt = poly_basis(xt, degree=0)
    f = saddle_point_predict(Kt, Pt, a, beta)

    max_err = np.max(np.abs(f - y.mean()))
    assert max_err <= 1e-4, f"max_err={max_err:.3e}"
