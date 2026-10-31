"""N-dimensional smoothing splines (Duchon / polyharmonic) with calibrated
posterior variance, plus the Matern -> polyharmonic flat-limit scaling law.

Background
----------
For d input dimensions and derivative order m with 2m > d, the Duchon
(thin-plate) smoothing spline minimises

    sum_i (y_i - f(x_i))^2 + lam * J_m(f),
    J_m(f) = int_{R^d} sum_{|a|=m} m!/a! (D^a f)^2 dx,

whose null space is the polynomials of total degree <= m-1 and whose
reproducing (generalised) kernel is the polyharmonic Green's function

    G(r) = a_{m,d} r^{2m-d}            (d odd)
    G(r) = a_{m,d} r^{2m-d} log r      (d even)

with the constant a_{m,d} chosen so that `nugget = lam` in the bordered
system of `uk.saddle_point_solve` reproduces EXACTLY the penalty lam*J_m
(no hidden rescaling).  Special cases:

    d=1, m=2 :  r^3 / 12                 == make_smoothing_spline(lam)
    d=2, m=2 :  r^2 log r / (8 pi)       == RBFInterpolator("thin_plate_spline",
                                                             smoothing=8*pi*lam)
    d=3, m=2 :  -r / (8 pi)              == RBFInterpolator("linear",
                                                             smoothing=8*pi*lam)

Bayesian reading (Kimeldorf & Wahba 1970; Wahba 1978, 1983): y = f + eps,
eps ~ N(0, sigma^2), f = (flat prior on null space) + sqrt(b) * Z, Z a
generalised GP with covariance G.  The posterior mean is the spline with
lam = sigma^2 / b; the posterior variance is

    Var[f(x*) | y] = b * ( G(0) - r*^T C^{-1} r* ),   b = sigma^2 / lam,

with C the bordered matrix.  sigma^2 (equivalently b) must be ESTIMATED --
the unit-scale variance `uk.saddle_point_variance` returns is not calibrated
on its own.

Matern flat limit (generalises the paper's 1-D Corollary 5): an ISOTROPIC
Matern-nu kernel with nu = m - d/2 and inverse length kappa satisfies

    k_nu(r; kappa) = (even polynomial in r) + c_nu kappa^{2 nu} * phi(r) + o(kappa^{2 nu})

where phi(r) = r^{2nu} (or r^{2nu} log r for integer nu), so universal
Kriging with trend Pi_{m-1} and nugget = (c_nu / a_{m,d}) kappa^{2 nu} lam
converges to the Duchon spline as kappa -> 0.  For d=1, nu=3/2 this is the
paper's `4 kappa^3 lam`.  A PRODUCT (tensor) Matern kernel instead converges
to an ADDITIVE spline (no interactions) -- see `test_hypotheses_2D_RANS`.
"""
from itertools import combinations_with_replacement
from math import comb

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma, kv

from .uk import saddle_point_solve, saddle_point_predict, saddle_point_variance


# ---------------------------------------------------------------------------
# Kernels and null space
# ---------------------------------------------------------------------------

def min_order(d):
    """Smallest admissible derivative order m for dimension d (2m > d)."""
    return d // 2 + 1


def null_space_dim(m, d):
    """dim Pi_{m-1}(R^d) = C(m-1+d, d): the minimum number of (unisolvent)
    data points needed before the spline is even defined."""
    return comb(m - 1 + d, d)


def duchon_constant(m, d):
    """a_{m,d} such that G = a_{m,d} r^{2m-d} [log r] is the Green's function
    of J_m (Duchon 1977; Wendland 2005, Thm 10.36 normalisation)."""
    if 2 * m <= d:
        raise ValueError(f"need 2m > d, got m={m}, d={d}")
    if d % 2 == 1:
        return gamma(d / 2 - m) / (2 ** (2 * m) * np.pi ** (d / 2) * gamma(m))
    k = m - d // 2
    return (-1) ** (k + 1) / (2 ** (2 * m - 1) * np.pi ** (d / 2) * gamma(m) * gamma(k + 1))


def _phi(r, m, d):
    p = 2 * m - d
    if d % 2 == 1:
        return r ** p
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r ** p * np.log(np.where(r > 0, r, 1.0)), 0.0)


def pairwise_dist(X1, X2):
    X1 = np.atleast_2d(np.asarray(X1, float))
    X2 = np.atleast_2d(np.asarray(X2, float))
    return np.sqrt(np.maximum(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1), 0.0))


def polyharmonic_kernel(X1, X2, m, d=None):
    """Duchon reproducing kernel G(||x-x'||), normalised so nugget = lam."""
    X1 = np.atleast_2d(np.asarray(X1, float))
    d = X1.shape[1] if d is None else d
    return duchon_constant(m, d) * _phi(pairwise_dist(X1, X2), m, d)


def poly_basis_nd(X, degree):
    """Monomials of total degree <= `degree` in d variables, shape (n, q)."""
    X = np.atleast_2d(np.asarray(X, float))
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for idx in combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, idx], axis=1))
    return np.column_stack(cols)


def matern(r, nu, kappa):
    """Isotropic Matern correlation, k(0)=1, parametrised as (kappa r)^nu K_nu(kappa r)
    (no sqrt(2 nu) factor, matching `kernels.matern32` for nu=3/2)."""
    x = kappa * np.asarray(r, float)
    with np.errstate(all="ignore"):
        v = 2 ** (1 - nu) / gamma(nu) * x ** nu * kv(nu, x)
    return np.where(x > 0, v, 1.0)


def matern_flat_coefficient(nu):
    """c_nu: coefficient of (kappa r)^{2nu} [log(kappa r)] in the small-r
    expansion of `matern(r, nu, kappa)`."""
    if float(nu).is_integer():
        nu = int(nu)
        return (-1) ** (nu + 1) * 2.0 / (2 ** (2 * nu) * gamma(nu) * gamma(nu + 1))
    return gamma(-nu) / (2 ** (2 * nu) * gamma(nu))


def matern_flat_nugget(kappa, lam, m, d):
    """Nugget making Matern-(m - d/2) universal Kriging converge to the Duchon
    spline with penalty lam*J_m as kappa -> 0.  d=1, m=2 gives 4 kappa^3 lam."""
    nu = m - d / 2
    return matern_flat_coefficient(nu) / duchon_constant(m, d) * kappa ** (2 * nu) * lam


# ---------------------------------------------------------------------------
# Smoothing spline with GCV / REML and calibrated variance
# ---------------------------------------------------------------------------

class PolyharmonicSmoothingSpline:
    """Duchon smoothing spline in d dimensions with calibrated posterior variance.

    Parameters
    ----------
    X : (n, d) inputs.  Scale them first if directions have different units --
        the penalty J_m is isotropic in whatever coordinates you pass.
    y : (n,) outputs.
    m : derivative order (default: smallest admissible, d//2 + 1).
    lam : float, or "gcv" / "reml" to select it (default "gcv").
    sigma2 : "reml" (default), "wahba" (RSS / tr(I - S)), or a known float.

    Methods: mean(Xq), variance(Xq) (of f, calibrated), predictive_variance
    (adds sigma^2), plus attributes lam, sigma2, b (= sigma2/lam), edf.
    """

    def __init__(self, X, y, m=None, lam="gcv", sigma2="reml", lam_grid=None):
        self.X = np.atleast_2d(np.asarray(X, float))
        self.y = np.asarray(y, float).reshape(-1)
        self.n, self.d = self.X.shape
        self.m = min_order(self.d) if m is None else int(m)
        self.q = null_space_dim(self.m, self.d)
        if self.n <= self.q:
            raise ValueError(f"n={self.n} <= null-space dim q={self.q} (m={self.m}, d={self.d})")

        self.K = polyharmonic_kernel(self.X, self.X, self.m, self.d)
        self.P = poly_basis_nd(self.X, self.m - 1)

        # Reduce to the space orthogonal to the null space: Q2^T K Q2 is SPD
        # for a conditionally positive definite kernel of order m.
        Q, _ = np.linalg.qr(self.P, mode="complete")
        Q2 = Q[:, self.q:]
        Kt = Q2.T @ self.K @ Q2
        self._dk, U = np.linalg.eigh(0.5 * (Kt + Kt.T))
        self._z = U.T @ (Q2.T @ self.y)

        if isinstance(lam, str):
            grid = np.logspace(-10, 4, 300) * np.median(np.abs(self._dk)) if lam_grid is None else lam_grid
            crit = self.gcv if lam == "gcv" else self.reml_objective
            scores = np.array([crit(l) for l in grid])
            i = int(np.nanargmin(scores))
            lo, hi = np.log10(grid[max(i - 1, 0)]), np.log10(grid[min(i + 1, len(grid) - 1)])
            res = minimize_scalar(lambda t: crit(10.0 ** t), bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-6})
            self.lam = float(10.0 ** res.x)
            self.lam_method = lam
        else:
            self.lam = float(lam)
            self.lam_method = "fixed"

        self.a, self.beta = saddle_point_solve(self.K, self.P, self.y, self.lam)
        self.edf = self.n - self._trace_I_minus_S(self.lam)

        if isinstance(sigma2, str):
            if sigma2 == "reml":
                self.b = self._b_reml(self.lam)
                self.sigma2 = self.b * self.lam
            elif sigma2 == "wahba":
                self.sigma2 = self._rss(self.lam) / self._trace_I_minus_S(self.lam)
                self.b = self.sigma2 / self.lam
            else:
                raise ValueError(sigma2)
        else:
            self.sigma2 = float(sigma2)
            self.b = self.sigma2 / self.lam

    # --- spectral helpers (O(n) per lambda after one eigendecomposition) ---
    def _trace_I_minus_S(self, lam):
        return np.sum(lam / (self._dk + lam))

    def _rss(self, lam):
        return np.sum((lam * self._z / (self._dk + lam)) ** 2)

    def _b_reml(self, lam):
        return np.mean(self._z ** 2 / (self._dk + lam))

    def gcv(self, lam):
        return self.n * self._rss(lam) / self._trace_I_minus_S(lam) ** 2

    def reml_objective(self, lam):
        nq = self.n - self.q
        return nq * np.log(self._b_reml(lam)) + np.sum(np.log(self._dk + lam))

    # --- prediction ---
    def mean(self, Xq):
        Xq = np.atleast_2d(np.asarray(Xq, float))
        return saddle_point_predict(polyharmonic_kernel(Xq, self.X, self.m, self.d),
                                    poly_basis_nd(Xq, self.m - 1), self.a, self.beta)

    def unit_variance(self, Xq):
        """Uncalibrated (b = 1) variance -- what `saddle_point_variance` returns."""
        Xq = np.atleast_2d(np.asarray(Xq, float))
        Kq = polyharmonic_kernel(Xq, self.X, self.m, self.d)
        Pq = poly_basis_nd(Xq, self.m - 1)
        return saddle_point_variance(self.K, self.P, self.lam, Kq, Pq, np.zeros(len(Xq)))

    def variance(self, Xq):
        """Calibrated posterior variance of f(x*)."""
        return np.maximum(self.b * self.unit_variance(Xq), 0.0)

    def predictive_variance(self, Xq):
        """Variance of a new noisy observation y* = f(x*) + eps."""
        return self.variance(Xq) + self.sigma2
