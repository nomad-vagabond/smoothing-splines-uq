"""Universal-Kriging / penalised-regression solver -- paper Theorem 2.

Two independent implementations of the SAME predictor are provided on
purpose:

  * `saddle_point_solve` / `saddle_point_predict` -- Theorem 2(a), the
    representer-theorem bordered system.
  * `gls_predict` -- Theorem 2(b), the explicit GLS / universal-Kriging
    formula, eliminating `a` analytically instead of solving the bordered
    system directly.

Phase A2 checks these agree to machine precision. Every later phase that
needs a Kriging/spline solve reuses `saddle_point_solve` so that a bug fixed
here is fixed everywhere downstream.
"""
import numpy as np


def poly_basis(x, degree):
    """Trend basis P = span{1, x, ..., x^degree}.

    x : (n,) -> P : (n, degree+1), columns [1, x, x^2, ...].
    """
    x = np.asarray(x).reshape(-1)
    return np.vstack([x**p for p in range(degree + 1)]).T


def saddle_point_solve(K, P, y, nugget, weights=None):
    """Theorem 2(a): solve

        [[K + nugget*Winv, P], [P^T, 0]] @ [a; beta] = [y; 0]

    where Winv = diag(1/weights) (weights=None => homoscedastic, Winv = I).
    `nugget` is the raw noise-to-signal ratio tau^2/sigma^2 (eq. 3), NOT a
    rescaled quantity -- callers are responsible for any Table-1-specific
    rescaling (see e.g. Phase B2's `4*kappa**3*lam` for Matern-3/2).

    Uses `lstsq` rather than `solve`: the bordered system is near-singular
    whenever the nugget is tiny and/or the kernel matrix is ill-conditioned
    (e.g. small theta/kappa), which happens on purpose in these checks.

    The system is diagonally equilibrated before the solve (A -> D^-1 A D^-1,
    D = sqrt(diag(|A|))) and un-scaled after. Without this, `lstsq`'s default
    `rcond` truncates real information whenever the nugget and kernel blocks
    span many orders of magnitude -- found in Phase A1 (nugget=1e8 was
    silently solved as beta=0 instead of the correct mean(y); equilibration
    fixes it through at least nugget=1e14).

    Returns (a, beta).
    """
    K = np.asarray(K)
    P = np.asarray(P)
    y = np.asarray(y).reshape(-1)
    n = K.shape[0]
    q = P.shape[1]
    Winv = np.eye(n) if weights is None else np.diag(1.0 / np.asarray(weights))

    A = np.zeros((n + q, n + q))
    A[:n, :n] = K + nugget * Winv
    A[:n, n:] = P
    A[n:, :n] = P.T
    rhs = np.zeros(n + q)
    rhs[:n] = y

    d = np.sqrt(np.abs(np.diag(A)))
    d[d == 0] = 1.0
    A_scaled = A / d[:, None] / d[None, :]
    rhs_scaled = rhs / d

    sol_scaled, *_ = np.linalg.lstsq(A_scaled, rhs_scaled, rcond=None)
    sol = sol_scaled / d
    return sol[:n], sol[n:]


def saddle_point_predict(Kstar, Pstar, a, beta):
    """f(x*) = P(x*)^T beta + k(x*,X)^T a."""
    return np.asarray(Pstar) @ beta + np.asarray(Kstar) @ a


def gls_predict(K, P, y, nugget, Kstar, Pstar, weights=None):
    """Theorem 2(b): explicit GLS / universal-Kriging predictor.

        M = K + nugget*Winv
        beta_hat = (P^T M^-1 P)^-1 P^T M^-1 y            (GLS estimator)
        f(x*)    = P(x*)^T beta_hat + k(x*,X)^T M^-1 (y - P beta_hat)

    Independent solve path from `saddle_point_solve` (no shared
    sub-expressions beyond K, P, y themselves) -- used in Phase A2 to check
    the two formulations of Theorem 2 against each other.

    Returns (f_star, beta_hat).
    """
    K = np.asarray(K)
    P = np.asarray(P)
    y = np.asarray(y).reshape(-1)
    n = K.shape[0]
    Winv = np.eye(n) if weights is None else np.diag(1.0 / np.asarray(weights))

    M = K + nugget * Winv
    Minv_y = np.linalg.solve(M, y)
    Minv_P = np.linalg.solve(M, P)
    PtMinvP = P.T @ Minv_P
    beta_hat = np.linalg.solve(PtMinvP, P.T @ Minv_y)

    resid = y - P @ beta_hat
    Minv_resid = np.linalg.solve(M, resid)
    f_star = np.asarray(Pstar) @ beta_hat + np.asarray(Kstar) @ Minv_resid
    return f_star, beta_hat


def saddle_point_variance(K, P, nugget, Kstar, Pstar, Kstarstar, weights=None):
    """Universal-Kriging / Theorem 2 predictive variance, sigma^2=1 units
    (matching `saddle_point_solve`'s `nugget = tau^2/sigma^2` convention --
    multiply the result by your own sigma^2 estimate if you need absolute
    units).

        Var[f(x*)] / sigma^2 = k(x*,x*) - [k(x*,X); P(x*)]^T C^-1 [k(x*,X); P(x*)]

    where `C` is the same bordered matrix `saddle_point_solve` builds. This
    is the Lagrange-multiplier/dual-kriging-weights form of the universal
    kriging variance: solving `C @ [lambda; mu] = [k(x*,X); P(x*)]` gives the
    kriging weights `lambda` and the trend's Lagrange multipliers `mu`, and
    the variance is `k(x*,x*) - [k(x*,X); P(x*)]^T [lambda; mu]` -- so it
    reuses `saddle_point_solve`'s bordered system with a different
    right-hand side (`[k(x*,X); P(x*)]` instead of `[y; 0]`), including the
    same diagonal equilibration (needed for the same reason: this system is
    near-singular whenever the nugget or kernel matrix spans many orders of
    magnitude).

    `Kstar`, `Pstar`, `Kstarstar` may hold several query points at once:
    `Kstar` is (n_query, n), `Pstar` is (n_query, q), `Kstarstar` is
    (n_query,) (i.e. k(x*,x*) per query point, not the full query-by-query
    Gram matrix). Returns an (n_query,) array.
    """
    K = np.asarray(K)
    P = np.asarray(P)
    n = K.shape[0]
    q = P.shape[1]
    Winv = np.eye(n) if weights is None else np.diag(1.0 / np.asarray(weights))

    A = np.zeros((n + q, n + q))
    A[:n, :n] = K + nugget * Winv
    A[:n, n:] = P
    A[n:, :n] = P.T

    Kstar = np.atleast_2d(Kstar)
    Pstar = np.atleast_2d(Pstar)
    rhs = np.vstack([Kstar.T, Pstar.T])  # (n+q, n_query)

    d = np.sqrt(np.abs(np.diag(A)))
    d[d == 0] = 1.0
    A_scaled = A / d[:, None] / d[None, :]
    rhs_scaled = rhs / d[:, None]

    sol_scaled, *_ = np.linalg.lstsq(A_scaled, rhs_scaled, rcond=None)
    sol = sol_scaled / d[:, None]

    quad = np.einsum("ij,ij->j", rhs, sol)
    return np.asarray(Kstarstar) - quad
