"""Phase A2 -- saddle-point solve === explicit GLS/UK predictor (Theorem 2b).

See "Research Plan - From Fundamentals to Gaussian RMTS.md", Phase A, A2.
This is an algebraic identity (both sides solve the exact same optimisation
problem), so it must hold to near machine precision, independent of any
asymptotic regime -- unlike Phase B, there is no "approximately" here.
"""
import numpy as np

from smoothing_splines_uq.kernels import abs_exp
from smoothing_splines_uq.uk import poly_basis, saddle_point_solve, saddle_point_predict, gls_predict


def test_a2_saddle_point_matches_explicit_gls_predictor():
    rng = np.random.default_rng(2)
    n = 6
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(3 * x) + 0.05 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, n)  # heteroscedastic: exercise W too
    theta = 1.3
    nugget = 0.07

    K = abs_exp(np.abs(x[:, None] - x[None, :]), theta)
    P = poly_basis(x, degree=1)  # linear trend

    a, beta = saddle_point_solve(K, P, y, nugget, weights=weights)

    xt = np.linspace(0, 1, 97)
    Kt = abs_exp(np.abs(xt[:, None] - x[None, :]), theta)
    Pt = poly_basis(xt, degree=1)
    f_saddle = saddle_point_predict(Kt, Pt, a, beta)

    f_gls, beta_gls = gls_predict(K, P, y, nugget, Kt, Pt, weights=weights)

    beta_err = np.max(np.abs(beta - beta_gls))
    f_err = np.max(np.abs(f_saddle - f_gls))
    assert beta_err <= 1e-10, f"beta_err={beta_err:.3e}"
    assert f_err <= 1e-10, f"f_err={f_err:.3e}"
