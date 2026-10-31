"""Polyharmonic (Duchon) smoothing splines: exact agreement with scipy,
the Matern flat-limit scaling law, and calibrated variance."""
import numpy as np
import pytest
from scipy.interpolate import make_smoothing_spline, RBFInterpolator

from smoothing_splines_uq.polyharmonic import (
    PolyharmonicSmoothingSpline, duchon_constant, matern, matern_flat_nugget,
    pairwise_dist, poly_basis_nd, null_space_dim,
)
from smoothing_splines_uq.uk import saddle_point_solve, saddle_point_predict


def test_constants_match_classical_kernels():
    assert np.isclose(duchon_constant(2, 1), 1 / 12)            # cubic spline r^3/12
    assert np.isclose(duchon_constant(2, 2), 1 / (8 * np.pi))   # thin plate r^2 log r / 8pi
    assert np.isclose(duchon_constant(2, 3), -1 / (8 * np.pi))  # 3-D biharmonic -r/8pi
    assert np.isclose(matern_flat_nugget(0.1, 1.0, m=2, d=1), 4 * 0.1**3)  # paper's 4 kappa^3


def test_1d_matches_make_smoothing_spline():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(6 * x) + 0.1 * rng.standard_normal(40)
    xd = np.linspace(x.min(), x.max(), 100)
    s = PolyharmonicSmoothingSpline(x[:, None], y, m=2, lam=1e-2)
    assert np.max(np.abs(s.mean(xd[:, None]) - make_smoothing_spline(x, y, lam=1e-2)(xd))) < 1e-9


@pytest.mark.parametrize("d,m,kernel,sign", [(2, 2, "thin_plate_spline", 1), (3, 2, "linear", -1),
                                              (3, 3, "cubic", 1), (5, 3, "linear", -1)])
def test_nd_matches_rbfinterpolator(d, m, kernel, sign):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (60, d))
    y = np.sin(X.sum(1)) + 0.05 * rng.standard_normal(60)
    Xq = rng.uniform(0, 1, (20, d))
    lam = 1e-3
    s = PolyharmonicSmoothingSpline(X, y, m=m, lam=lam)
    ref = RBFInterpolator(X, y, kernel=kernel, smoothing=lam * sign / duchon_constant(m, d), degree=m - 1)(Xq)
    assert np.max(np.abs(s.mean(Xq) - ref)) < 1e-9


@pytest.mark.parametrize("d,m", [(1, 2), (2, 2), (3, 2)])
def test_isotropic_matern_flat_limit(d, m):
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (40, d))
    y = np.cos(2 * X.sum(1))
    Xq = rng.uniform(0, 1, (20, d))
    lam = 1e-3
    target = PolyharmonicSmoothingSpline(X, y, m=m, lam=lam).mean(Xq)
    P, Pq = poly_basis_nd(X, m - 1), poly_basis_nd(Xq, m - 1)
    errs = []
    for k in [0.3, 0.03]:
        nu = m - d / 2
        a, b = saddle_point_solve(matern(pairwise_dist(X, X), nu, k), P, y, matern_flat_nugget(k, lam, m, d))
        errs.append(np.max(np.abs(saddle_point_predict(matern(pairwise_dist(Xq, X), nu, k), Pq, a, b) - target)))
    assert errs[1] < errs[0] / 5


def test_gcv_and_calibrated_variance():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (120, 2))
    f = lambda Z: np.sin(3 * Z[:, 0]) * np.cos(2 * Z[:, 1])
    y = f(X) + 0.1 * rng.standard_normal(120)
    s = PolyharmonicSmoothingSpline(X, y, lam="gcv", sigma2="wahba")
    assert 0.05 < np.sqrt(s.sigma2) < 0.2
    Xq = rng.uniform(0.1, 0.9, (200, 2))
    cover = np.mean(np.abs(s.mean(Xq) - f(Xq)) <= 1.96 * np.sqrt(s.variance(Xq)))
    assert cover > 0.8
    assert null_space_dim(s.m, 2) == 3
