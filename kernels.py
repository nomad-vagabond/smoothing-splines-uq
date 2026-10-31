"""Kernel functions and the kernel <-> energy correspondence (paper Table 1).

Only the pieces Phase A needs: the Matern-1/2 (`abs_exp`) correlation used in
A3, and the Brownian/Wiener kernel used in A1/A2 as a hand-checkable, exact
RKHS (its native penalty is \\int h'^2, i.e. the kernel whose minimal-norm
interpolant and whose penalised-regression limit are both the natural LINEAR
spline -- Table 1's "degree 1" row). Matern-3/2 and 5/2 (needed from Phase B
onward) are included too since they cost nothing extra here.
"""
import numpy as np


def abs_exp(d, theta):
    """Matern-1/2 correlation (SMT's `abs_exp`): k(d) = exp(-theta*|d|)."""
    return np.exp(-theta * np.abs(d))


def matern32(d, kappa):
    """Matern-3/2 correlation (SMT's `matern32`): (1+kappa|d|) exp(-kappa|d|)."""
    ad = kappa * np.abs(d)
    return (1.0 + ad) * np.exp(-ad)


def matern52(d, kappa):
    """Matern-5/2 correlation (SMT's `matern52`)."""
    ad = kappa * np.abs(d)
    return (1.0 + ad + ad**2 / 3.0) * np.exp(-ad)


def brownian(x1, x2):
    """k(x,x') = min(x,x') -- the Wiener-process kernel.

    Its RKHS norm is exactly \\int h'^2 subject to h(0)=0, so the minimal-norm
    interpolant (nugget=0) and the penalised-regression limit under this
    kernel are both the natural piecewise-LINEAR spline, with no unknown
    normalisation constant to calibrate. Used in Phase A1/A2 as ground truth
    that does not depend on the Matern/Table-1 scaling derivation used in A3.
    """
    return np.minimum(x1, x2)


def iwp2_kernel(x1, x2):
    """Reproducing kernel for the m=2 (cubic) derivative-penalty functional
    \\int h''^2, modulo the degree-1 null space {1, x} (trend
    `poly_basis(x, degree=1)`) -- the covariance of twice-integrated Brownian
    motion (Kimeldorf & Wahba, 1970):

        G(x,x') = min(x,x')^2 * (3*max(x,x') - min(x,x')) / 6

    Table 1's cubic-spline row (m=2, degree 3), without the Matern/kappa
    detour Corollary 5 uses: plugged into `saddle_point_solve` with
    `nugget=lambda` directly (no `4*kappa**3` rescaling -- there is no kappa
    here), this reproduces `scipy.interpolate.make_smoothing_spline(lam=
    lambda)`'s mean to ~1e-12, and `saddle_point_variance` with the same `K`
    gives the companion predictive variance -- see
    `notebooks/spline_variance_without_kriging.ipynb`.

    Like `brownian`, this kernel implicitly anchors at x=0 (it is the
    covariance of a process started at 0); shift x close to 0 first for good
    conditioning if the domain is far from the origin.
    """
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    return lo**2 * (3.0 * hi - lo) / 6.0


def tps2d_kernel(X1, X2):
    """Thin-plate-spline reproducing kernel (Duchon, 1977), the 2-D analogue
    of `iwp2_kernel`: reproducing kernel for the m=2 bending-energy penalty
    \\iint[(f_xx)^2 + 2 f_xy^2 + (f_yy)^2] dx dy, modulo the degree-1 null
    space {1, x, y} (trend `poly_basis_2d`):

        K(x,x') = ||x-x'||^2 * log(||x-x'||),   K(x,x) = 0 by convention.

    Unlike `iwp2_kernel` this needs no anchor: the thin-plate kernel is
    translation- *and* rotation-invariant, so it has no implicit "started at
    0" boundary condition to worry about.

    Plugged into `saddle_point_solve` with `nugget=lambda` directly (no
    `kappa`, same as `iwp2_kernel`), this reproduces
    `scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline",
    smoothing=lambda, degree=1)`'s mean to ~1e-14 -- scipy's own docstring
    for `RBFInterpolator` writes out this exact bordered system (Theorem 2a)
    and cites Wahba (1990) directly. `saddle_point_variance` with the same
    `K` gives the companion predictive variance scipy does not expose --
    see `notebooks/spline_variance_2d_vs_smt.ipynb`.

    `X1`, `X2` : (n1, 2), (n2, 2) arrays of point coordinates (point-set
    input, unlike the 1-D kernels above which take pre-broadcast arrays --
    pairwise Euclidean distance in 2-D is awkward to broadcast by hand at
    the call site, so this function does it internally). Returns an
    (n1, n2) matrix.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    d = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(d > 0, d**2 * np.log(d), 0.0)
    return K


def poly_basis_2d(X):
    """Trend P = span{1, x, y} for 2-D inputs -- the `tps2d_kernel` null
    space, the degree-1 analogue of `poly_basis(x, degree=1)`.

    X : (n, 2) -> P : (n, 3), columns [1, x, y].
    """
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1]])


def pairwise_abs_diff(x1, x2):
    """|x_i - x_j| for 1-D inputs, broadcasting x1 as column, x2 as row."""
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    return np.abs(x1 - x2)
