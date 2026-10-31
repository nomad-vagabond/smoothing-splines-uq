"""Phase A3 -- cheapest instance of the kernel<->energy map (Table 1):
Matern-1/2 <-> linear spline.

See "Research Plan - From Fundamentals to Gaussian RMTS.md", Phase A, A3.

Table 1 gives the energy "up to constants" and only works out the explicit
kappa-scaling for the cubic (Matern-3/2) case in Corollary 5 (nugget =
4*kappa^3*lambda). For the linear case (Matern-1/2, m=1) the same spectral
argument (sketched in `_gp_mean`'s docstring below) gives nugget ~ theta^1,
one power lower than the cubic case's theta^3 -- consistent with "one fewer
derivative in the energy -> one lower power of the length-scale parameter".

Because the overall multiplicative constant is left unconstrained by the
paper ("up to constants"), both checks below are constant-agnostic:
  (1) self-convergence rate: compare the sweep against its own finest-theta
      member, and fit the log-log slope (should be near the predicted
      power, 1, not the cubic case's ~1.5-2);
  (2) structural check: confirm the limiting curve is actually piecewise
      LINEAR between consecutive data points (Table 1's claimed limit is
      "linear spline, C^0"), not merely that the sweep converges to
      *something*.
"""
import numpy as np

from smoothing_splines_uq.kernels import abs_exp
from smoothing_splines_uq.uk import poly_basis, saddle_point_solve, saddle_point_predict


def _gp_mean(x, y, theta, lam_ref, xt):
    """Matern-1/2 universal-Kriging mean with nugget(theta) = theta * lam_ref.

    Derivation of the theta^1 scaling (not in the paper): the Matern-nu
    covariance has spectral density S(w) = sigma^2 * N_m * kappa^(2m-1) *
    (kappa^2+w^2)^(-m), the kappa^(2m-1) factor being required so that the
    variance sigma^2 = \\int S dw stays independent of kappa. Inverting S and
    expanding (kappa^2+w^2)^m, the term surviving as kappa->0 is the top
    derivative one, with prefactor kappa^(1-2m) * \\int (h^(m))^2. For m=2
    (cubic) this reproduces the paper's own kappa^3 nugget law (Phase B2).
    For m=1 (linear, this file) it gives kappa^(1-2) = kappa^(-1), i.e.
    nugget ~ theta^1.
    """
    K = abs_exp(np.abs(x[:, None] - x[None, :]), theta)
    P = poly_basis(x, degree=0)  # trend = span{1}, matches Table 1's row
    nugget = theta * lam_ref
    a, beta = saddle_point_solve(K, P, y, nugget)
    Kt = abs_exp(np.abs(xt[:, None] - x[None, :]), theta)
    Pt = poly_basis(xt, degree=0)
    return saddle_point_predict(Kt, Pt, a, beta)


def _benchmark(rng):
    n = 40
    x = np.linspace(0, 1, n)
    y = np.sin(6 * x) + 0.1 * rng.standard_normal(n)
    return x, y


def test_a3_self_convergence_rate_matches_predicted_power():
    """The local (pairwise) convergence rate is NOT a single power across the
    whole sweep: starting from theta=10 down to theta=0.003, the local slope
    drifts from ~2.0 down to ~1.0 (checked by hand before writing this
    tolerance in -- see the research log). That is a real pre-asymptotic
    effect (at large theta the GP is nowhere near the theta->0 regime this
    whole check is about), not sweep noise, so the fit below is restricted to
    the small-theta end where the local slope has actually settled, instead
    of being averaged across both regimes and landing on a biased ~1.6-1.7."""
    rng = np.random.default_rng(3)
    x, y = _benchmark(rng)
    xt = np.linspace(0, 1, 200)
    lam_ref = 1e-4

    thetas = np.array([0.3, 0.1, 0.03, 0.01])
    f_ref = _gp_mean(x, y, 0.001, lam_ref, xt)  # well below the sweep

    errs = np.array([
        np.max(np.abs(_gp_mean(x, y, th, lam_ref, xt) - f_ref)) for th in thetas
    ])

    assert np.all(np.diff(errs) < 0), f"errors not monotone decreasing: {errs}"

    slope, _ = np.polyfit(np.log(thetas), np.log(errs), 1)
    assert 0.8 <= slope <= 1.3, f"slope={slope:.3f}, errs={errs}"


def test_a3_limit_is_structurally_a_linear_spline():
    rng = np.random.default_rng(3)
    x, y = _benchmark(rng)
    lam_ref = 1e-4
    theta = 0.01

    xt = np.linspace(0, 1, 2001)
    f = _gp_mean(x, y, theta, lam_ref, xt)

    f_at_knots = np.interp(x, xt, f)
    chord = np.interp(xt, x, f_at_knots)
    max_dev = np.max(np.abs(f - chord))

    rel_dev = max_dev / np.ptp(f)
    assert rel_dev <= 0.02, f"rel_dev={rel_dev:.4f} (max_dev={max_dev:.3e})"
