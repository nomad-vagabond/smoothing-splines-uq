"""A third, independent RMTB-variance cross-check -- for validation
notebooks, not the main package API (see `rmtb_uq.RMTBUQ` for that).

This is a generic Bayesian-ridge-regression UQ tool: given *any* basis
design matrix `B` and penalty matrix `K`, it fits `c_hat = (B^T B + K)^-1
B^T y` and the matching posterior covariance. It has no connection to RMTB
on its own -- a contributed version of this module, left unwired to any real
model, was checked against `smt.surrogate_models.RMTB` on identical data and
found to diverge substantially (max mean difference `0.61` on a target range
of `~[-1.5,1.5]`), because its own demo built a *different* B-spline basis
and a *different* (finite-difference, Eilers-Marx/P-spline style) penalty
than RMTB's actual quadrature-integrated bending-energy structure.

**Its real, validated value**: once `B` and `K` are explicitly wired to a
trained `RMTB` model's actual `Phi` (`full_jac_dict[0]`) and `H`
(`full_hess`), `fit()`'s `c_hat` reproduces `model.sol` to `~1e-11`, and
`predict()`'s variance reproduces `rmtb_uq.RMTBUQ.predict_variances()` to
`~1e-18` -- a genuine fifth independent code path for the same validated
`Var_bayes` formula, agreeing because the underlying math is the same, not
because any code is shared. See
`07_phaseD_rmtb_variance.ipynb`'s "Cross-check 3" for the full wiring and
the numbers above.

This also found a real, worth-knowing robustness gap: without the explicit
`cond()` check added below, `scipy.linalg.solve(..., assume_a='pos')` does
*not* raise on a near-singular `G_reg` -- it returns a garbage answer with
only a `LinAlgWarning` (easy to miss) as any indication something went
wrong. Fixed here by checking `cond(G_reg)` against this project's `1e8`
safety line before trusting the solve, rather than silently returning a
number. The original (un-checked) behavior is why this
class is validation-notebook-only, not part of the main public API: a tool
used for cross-checking needs to fail loudly when it can't be trusted,
which this version now does.
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import norm


class RMTBUncertaintyWrapper:
    """Post-hoc Bayesian-ridge UQ wrapper: `c_hat = (B^T B + K)^-1 B^T y`,
    `Cov(c_hat) = sigma^2 (B^T B + K)^-1` (`Cov=sigma^2*A` with `A=(B^TB+K)^-1`,
    Research Plan Section 2.7).

    Parameters
    ----------
    ridge_alpha : float
        Extra `ridge_alpha * I` added on top of `B^T B + K` before solving.
        Not RMTB's own `regularization_weight` -- a separate knob, on top of
        whatever `K` already carries. Keep small (`<=1e-8`) unless
        deliberately testing its effect -- the `cond()`-vs-`K_kappa`-scale
        interaction means "small" isn't always obviously small enough.
    cond_limit : float
        Safety line for `cond(B^T B + K + ridge_alpha*I)` (default `1e8`).
        `fit()` raises `RuntimeError` rather than silently solving an
        untrustworthy system above this.
    """

    def __init__(self, ridge_alpha: float = 1e-8, cond_limit: float = 1e8):
        self.ridge_alpha = ridge_alpha
        self.cond_limit = cond_limit
        self.c_hat = None
        self.Sigma_c = None
        self.sigma2 = None
        self.edf = None
        self._cond_G = None

    def fit(self, B: np.ndarray, y: np.ndarray, K: np.ndarray):
        """Fit control parameters and the posterior parameter covariance.

        Parameters
        ----------
        B : (N, D) basis/design matrix at N training points.
        y : (N,) observed target values.
        K : (D, D) penalty matrix (e.g. a trained RMTB model's `full_hess`,
            for the validated cross-check use -- see module docstring).
        """
        N, D = B.shape
        y = y.ravel()

        G = B.T @ B + K
        G_reg = G + self.ridge_alpha * np.eye(D)

        self._cond_G = np.linalg.cond(G_reg)
        if self._cond_G > self.cond_limit:
            raise RuntimeError(
                f"cond(B^T B + K + ridge_alpha*I) = {self._cond_G:.3e} exceeds "
                f"cond_limit={self.cond_limit:.0e} -- refusing to solve a system "
                "this unsafe silently. scipy.linalg.solve would not raise here on "
                "its own (confirmed directly); it would return a garbage answer "
                "with only an easy-to-miss LinAlgWarning."
            )

        self.c_hat = la.solve(G_reg, B.T @ y, assume_a="pos")

        residuals = y - B @ self.c_hat
        G_inv_BtB = la.solve(G_reg, B.T @ B, assume_a="pos")
        self.edf = np.trace(G_inv_BtB)
        dof = max(1.0, N - self.edf)
        self.sigma2 = np.sum(residuals**2) / dof

        try:
            G_inv = la.inv(G_reg, overwrite_a=False)
        except la.LinAlgError:
            G_inv = la.pinv(G_reg)
        self.Sigma_c = self.sigma2 * G_inv
        return self

    def predict(self, B_new: np.ndarray, alpha: float = 0.05):
        """Predict mean, epistemic (curve-shape) variance, and aleatoric
        (new-observation) variance at new basis rows `B_new`.

        Returns
        -------
        dict containing 'mean', 'epistemic_var', 'aleatoric_var',
        'predictive_var', 'conf_interval', 'pred_interval'.
        """
        z_crit = norm.ppf(1.0 - alpha / 2.0)

        f_mean = B_new @ self.c_hat
        epistemic_var = np.einsum("ni,ij,nj->n", B_new, self.Sigma_c, B_new)
        epistemic_var = np.maximum(0.0, epistemic_var)
        aleatoric_var = self.sigma2
        total_predictive_var = epistemic_var + aleatoric_var

        epistemic_std = np.sqrt(epistemic_var)
        predictive_std = np.sqrt(total_predictive_var)

        return {
            "mean": f_mean,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "predictive_var": total_predictive_var,
            "conf_interval": (f_mean - z_crit * epistemic_std, f_mean + z_crit * epistemic_std),
            "pred_interval": (f_mean - z_crit * predictive_std, f_mean + z_crit * predictive_std),
        }
