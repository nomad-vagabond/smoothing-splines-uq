"""`SmoothingSpline1D`/`SmoothingSpline2D`: the
mean must be bit-identical to the scipy object each class wraps, and the
variance must be well-behaved (finite, non-negative, larger away from data)
even though scipy itself exposes no ground truth to compare it against.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline, RBFInterpolator

from smoothing_splines_uq.smoothing_spline import SmoothingSpline1D, SmoothingSpline2D


def test_smoothing_spline_1d_mean_matches_scipy_exactly():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(6 * x) + 0.1 * rng.standard_normal(40)
    x_dense = np.linspace(x.min(), x.max(), 100)
    lam = 1e-2

    wrapped = SmoothingSpline1D(x, y, lam)
    target = make_smoothing_spline(x, y, lam=lam)(x_dense)

    assert np.max(np.abs(wrapped.mean(x_dense) - target)) < 1e-9


def test_smoothing_spline_1d_variance_well_behaved():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(6 * x) + 0.1 * rng.standard_normal(40)
    x_dense = np.linspace(-0.2, 1.2, 100)  # includes extrapolated region
    wrapped = SmoothingSpline1D(x, y, lam=1e-2)

    var = wrapped.variance(x_dense)
    assert np.all(var > -1e-10)

    in_domain = (x_dense > 0.1) & (x_dense < 0.9)
    assert var[~in_domain].mean() > var[in_domain].mean()


def test_smoothing_spline_2d_mean_matches_scipy_exactly():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(30, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + 0.05 * rng.standard_normal(30)
    Xq = rng.uniform(0, 1, size=(25, 2))
    lam = 1e-2

    wrapped = SmoothingSpline2D(X, y, lam)
    target = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam,
                              degree=1)(Xq)

    assert np.max(np.abs(wrapped.mean(Xq) - target)) < 1e-9


def test_smoothing_spline_2d_variance_well_behaved():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(30, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + 0.05 * rng.standard_normal(30)
    wrapped = SmoothingSpline2D(X, y, lam=1e-2)

    grid = np.linspace(-0.3, 1.3, 15)
    GX, GY = np.meshgrid(grid, grid)
    Xq = np.column_stack([GX.ravel(), GY.ravel()])

    var = wrapped.variance(Xq)
    assert np.all(var > -1e-10)

    in_domain = (Xq[:, 0] > 0.1) & (Xq[:, 0] < 0.9) & (Xq[:, 1] > 0.1) & (Xq[:, 1] < 0.9)
    assert var[~in_domain].mean() > var[in_domain].mean()
