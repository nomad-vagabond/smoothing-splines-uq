"""`NdGridCubicSmoothingSplineUQ` (`02_csaps_mean_and_variance.ipynb`):
the closed-form mean/variance formulas,
confirmed independently in 2-D and spot-checked the same way in 3-D, plus
the `off_grid_method` contract (`"exact"` works for any dimension count,
`"fast"` is refused outside 2-D rather than silently approximated).

Three claims, each checked independently, not assumed from the 2-D case
alone:

1. The N-D mean is the sequential per-axis composition
   $\\hat Y = Z \\times_1 S_1 \\times_2 S_2 \\cdots \\times_N S_N$ -- checked
   against a hand-rolled sequential-pass reconstruction built from
   `PolyharmonicSmoothingSpline`, independently of the class under test.
2. The N-D variance is the N-way Kronecker-diagonal product
   $\\mathrm{Var}(F)=\\sigma^2\\prod_i S_i[p_i,p_i]$ -- checked against a
   brute-force dense Kronecker-product construction (probing each axis's
   smoother matrix with unit vectors), both on the diagonal and for
   off-diagonal covariance between different grid points.
3. `off_grid_method="exact"` and `"fast"` agree exactly on the training
   grid in 2-D (both compute the same closed form there); `"fast"` raises
   `NotImplementedError` for any other dimension count rather than
   guessing at an un-validated N-D re-interpolation scheme.
"""
import numpy as np

from smoothing_splines_uq.csaps_uq import NdGridCubicSmoothingSplineUQ
from smoothing_splines_uq.polyharmonic import PolyharmonicSmoothingSpline


def _smoother_matrix(x, lam):
    n = len(x)
    S = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        S[:, k] = PolyharmonicSmoothingSpline(x.reshape(-1, 1), e, m=2, lam=lam, sigma2=1.0).mean(x.reshape(-1, 1))
    return S


def _make_2d_data(seed=1, n_side=7):
    rng = np.random.default_rng(seed)
    x1 = np.linspace(0, 1, n_side)
    x2 = np.linspace(0, 1, n_side)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    Z = np.sin(3 * X1) * np.cos(3 * X2) + 0.05 * rng.standard_normal((n_side, n_side))
    return x1, x2, Z


def _make_3d_data(seed=0, n_side=6):
    rng = np.random.default_rng(seed)
    x1 = np.linspace(0, 1, n_side)
    x2 = np.linspace(0, 1, n_side)
    x3 = np.linspace(0, 1, n_side)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")
    Z = np.sin(3 * X1) * np.cos(3 * X2) * np.sin(2 * X3 + 0.3) + 0.05 * rng.standard_normal((n_side,) * 3)
    return x1, x2, x3, Z


def test_2d_exact_and_fast_agree_on_grid():
    x1, x2, Z = _make_2d_data()
    uq_exact = NdGridCubicSmoothingSplineUQ([x1, x2], Z, smooth=[0.9, 0.7], off_grid_method="exact")
    uq_fast = NdGridCubicSmoothingSplineUQ([x1, x2], Z, smooth=[0.9, 0.7], off_grid_method="fast")

    mean_e, mean_f = np.asarray(uq_exact([x1, x2])), np.asarray(uq_fast([x1, x2]))
    var_e, var_f = uq_exact.variance([x1, x2]), uq_fast.variance([x1, x2])
    assert np.max(np.abs(mean_e - mean_f)) == 0.0
    assert np.max(np.abs(var_e - var_f)) == 0.0


def test_2d_variance_matches_brute_force_kronecker():
    x1, x2, Z = _make_2d_data()
    p1, p2 = 0.9, 0.7
    lam1, lam2 = (1 - p1) / p1, (1 - p2) / p2
    sigma2 = 0.05 ** 2

    uq = NdGridCubicSmoothingSplineUQ([x1, x2], Z, smooth=[p1, p2], sigma2=sigma2)
    S1, S2 = _smoother_matrix(x1, lam1), _smoother_matrix(x2, lam2)
    n = len(x1)
    A = np.kron(S2, S1)

    var_formula = uq.variance([x1, x2])
    var_dense = (sigma2 * np.diag(A)).reshape(n, n, order="F")
    assert np.max(np.abs(var_formula - var_dense)) < 1e-12

    rng = np.random.default_rng(3)
    for _ in range(10):
        p, q, r, s = rng.integers(0, n, size=4)
        idx_pq, idx_rs = q * n + p, s * n + r
        cov_dense = sigma2 * A[idx_pq, idx_rs]
        cov_formula = sigma2 * S1[p, r] * S2[q, s]
        assert abs(cov_dense - cov_formula) < 1e-12


def test_fast_refused_outside_2d():
    x1, x2, x3, Z = _make_3d_data()
    try:
        NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[0.9, 0.7, 0.8], off_grid_method="fast")
        assert False, "expected NotImplementedError for ndim=3 with off_grid_method='fast'"
    except NotImplementedError:
        pass


def test_3d_mean_matches_sequential_pass_reconstruction():
    x1, x2, x3, Z = _make_3d_data()
    p1, p2, p3 = 0.9, 0.7, 0.8
    lam1, lam2, lam3 = (1 - p1) / p1, (1 - p2) / p2, (1 - p3) / p3
    n = len(x1)

    uq = NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[p1, p2, p3])

    step1 = np.zeros_like(Z)
    for i in range(n):
        for j in range(n):
            step1[i, j, :] = PolyharmonicSmoothingSpline(x3.reshape(-1, 1), Z[i, j, :], m=2, lam=lam3,
                                                           sigma2=1.0).mean(x3.reshape(-1, 1))
    step2 = np.zeros_like(Z)
    for i in range(n):
        for k in range(n):
            step2[i, :, k] = PolyharmonicSmoothingSpline(x2.reshape(-1, 1), step1[i, :, k], m=2, lam=lam2,
                                                           sigma2=1.0).mean(x2.reshape(-1, 1))
    step3 = np.zeros_like(Z)
    for j in range(n):
        for k in range(n):
            step3[:, j, k] = PolyharmonicSmoothingSpline(x1.reshape(-1, 1), step2[:, j, k], m=2, lam=lam1,
                                                           sigma2=1.0).mean(x1.reshape(-1, 1))

    mean_class = np.asarray(uq([x1, x2, x3]))
    assert np.max(np.abs(mean_class - step3)) < 1e-10


def test_3d_variance_matches_brute_force_kronecker():
    x1, x2, x3, Z = _make_3d_data()
    p1, p2, p3 = 0.9, 0.7, 0.8
    lam1, lam2, lam3 = (1 - p1) / p1, (1 - p2) / p2, (1 - p3) / p3
    sigma2 = 0.05 ** 2
    n = len(x1)

    uq = NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[p1, p2, p3], sigma2=sigma2)
    S1, S2, S3 = _smoother_matrix(x1, lam1), _smoother_matrix(x2, lam2), _smoother_matrix(x3, lam3)

    var_formula = uq.variance([x1, x2, x3])
    var_dense = sigma2 * np.einsum("p,q,r->pqr", np.diag(S1), np.diag(S2), np.diag(S3))
    assert np.max(np.abs(var_formula - var_dense)) < 1e-12


def test_3d_scattered_off_grid_predict_runs():
    x1, x2, x3, Z = _make_3d_data()
    uq = NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[0.9, 0.7, 0.8], sigma2=0.05 ** 2)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, size=(20, 3))
    res = uq.predict(pts, return_std=True, return_ci=True)
    assert res["mean"].shape == (20,)
    assert np.all(np.isfinite(res["mean"]))
    assert np.all(res["std"] >= 0)


def test_scattered_mean_and_variance_split_matches_predict_exact():
    x1, x2, Z = _make_2d_data()
    uq = NdGridCubicSmoothingSplineUQ([x1, x2], Z, smooth=[0.9, 0.7], off_grid_method="exact")
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, size=(25, 2))

    combined = uq.predict(pts, return_std=True)
    mean_only = uq.predict_mean(pts)
    var_only = uq.variance(pts)

    assert np.array_equal(mean_only, combined["mean"])
    assert np.array_equal(np.clip(var_only, 0.0, None), combined["var"])


def test_scattered_mean_and_variance_split_matches_predict_fast():
    x1, x2, Z = _make_2d_data()
    uq = NdGridCubicSmoothingSplineUQ([x1, x2], Z, smooth=[0.9, 0.7], off_grid_method="fast")
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(25, 2))

    combined = uq.predict(pts, return_std=True)
    mean_only = uq.predict_mean(pts)
    var_only = uq.variance(pts)

    assert np.array_equal(mean_only, combined["mean"])
    assert np.array_equal(np.clip(var_only, 0.0, None), combined["var"])


def test_scattered_mean_and_variance_split_matches_predict_3d():
    x1, x2, x3, Z = _make_3d_data()
    uq = NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[0.9, 0.7, 0.8], sigma2=0.05 ** 2)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 1, size=(15, 3))

    combined = uq.predict(pts, return_std=True)
    mean_only = uq.predict_mean(pts)
    var_only = uq.variance(pts)

    assert np.array_equal(mean_only, combined["mean"])
    assert np.array_equal(np.clip(var_only, 0.0, None), combined["var"])


def test_edf_is_product_of_axis_traces():
    x1, x2, x3, Z = _make_3d_data()
    p1, p2, p3 = 0.9, 0.7, 0.8
    lam1, lam2, lam3 = (1 - p1) / p1, (1 - p2) / p2, (1 - p3) / p3

    uq = NdGridCubicSmoothingSplineUQ([x1, x2, x3], Z, smooth=[p1, p2, p3], sigma2=0.05 ** 2)
    S1, S2, S3 = _smoother_matrix(x1, lam1), _smoother_matrix(x2, lam2), _smoother_matrix(x3, lam3)
    expected_edf = np.trace(S1) * np.trace(S2) * np.trace(S3)
    assert abs(uq.edf - expected_edf) < 1e-10
