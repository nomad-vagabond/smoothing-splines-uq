"""SMT's `RMTB` extended with Bayesian posterior variance.

Merges what were two independently-developed exploratory implementations
into one. The two differed in both architecture and performance:

- `rmtb_uq.py` (subclass of `RMTB`, overriding `_train()`) -- a natural fit
  if this is ever proposed upstream to SMT itself (its other models already
  expose `predict_variances()` as part of the base `SurrogateModel`
  convention; a subclass that adds the same method is a small, reviewable
  diff). Its original per-query variance loop (sparse SuperLU factorization,
  one solve per point) was measured to be `6x-8x` slower than the
  alternative below (`10_variance_prediction_speed_benchmark.ipynb`).
- `rmtb_grid_uq.py` (composition around a plain `RMTB`) -- a single dense
  Cholesky factorization of `mtx`, solved in one batched call against every
  query point's basis row at once. Faster, but a parallel wrapper class
  rather than a natural SMT extension.

This class keeps the first architecture (subclassing, for exactly the reason
above) with the second's internals (the fast, batched Cholesky solve).

Both prior implementations were cross-validated against each other to
machine precision; nothing about the underlying math changed in this merge,
only which code computes it.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from smt.surrogate_models import RMTB


class RMTBUQ(RMTB):
    """`RMTB`, with `Var_bayes(x*) = sigma^2 * f(x*)^T mtx^-1 f(x*)` added
    (the weight-space posterior variance, Research Plan Section 2.7's
    `Cov=sigma^2*A` identity with `A=mtx^-1`).

    Parameters
    ----------
    sigma2 : "wahba" (default) or a known float
        Passed at construction; `"wahba"` estimates sigma^2 after training
        via the GCV-style `RSS/(n-edf)` estimator.
    **kwargs
        Passed through to `RMTB.__init__` unchanged.
    """

    def __init__(self, sigma2="wahba", **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = dict(kwargs)
        self._sigma2_arg = sigma2
        self.sigma2 = None
        self.edf = None
        self._mtx_chol = None

    def _train(self):
        super()._train()

        X_train, y_train_col = self.training_points[None][0]
        y_train = y_train_col.ravel()
        n = X_train.shape[0]

        mtx = self.mtx.toarray()
        self._mtx_chol = cho_factor(mtx, lower=True)

        Phi, _PhiT, _c = self.full_jac_dict[0]
        Phi = Phi.toarray()
        # Effective degrees of freedom: tr(Phi @ mtx^-1 @ Phi^T) ==
        # tr(mtx^-1 @ Phi^T @ Phi) by the cyclic property of trace.
        self.edf = float(np.trace(cho_solve(self._mtx_chol, Phi.T @ Phi)))

        if self._sigma2_arg == "wahba":
            resid = y_train - self._predict_values(X_train).ravel()
            self.sigma2 = float(resid @ resid) / max(n - self.edf, 1e-8)
        else:
            self.sigma2 = float(self._sigma2_arg)

    def _basis_row(self, x):
        f_mtx = self._compute_prediction_mtx(np.asarray(x, float), 0)
        return np.asarray(f_mtx.toarray() if hasattr(f_mtx, "toarray") else f_mtx)

    def predict_variances(self, x):
        """Weight-space posterior variance -- the recommended default route.
        One batched Cholesky solve for every query point at once."""
        F = self._basis_row(x)
        V = cho_solve(self._mtx_chol, F.T)
        var = self.sigma2 * np.sum(F * V.T, axis=1)
        return np.clip(var, 0.0, None)

    def predict_variances_dual(self, x, regularization_boost=1e-6):
        """Proposition 6's dual/kernel-space route -- a structurally
        independent cross-check of `predict_variances()`, NOT a recommended
        default. See `07_phaseD_rmtb_variance.ipynb` ("Step 5a, revisited")
        and `08_phaseD_step6_harder_case.ipynb` for the full derivation,
        validation, and caveats.

        RMTB's own `regularization_weight`, at its usual default, leaves
        `full_hess` too near-singular to invert directly (a genuine null
        space -- the unpenalized polynomial/bilinear trend -- not just
        poor conditioning). This method
        therefore refits a *separate*, regularization-boosted sibling model
        on the same training data to build the dual route, rather than using
        `self.full_hess` directly -- the boosted model's `sigma2`/`edf` are
        not used, only its `full_hess`/`full_jac_dict`; `self.sigma2` (this
        instance's own) is used throughout, matching how this project always
        compared the two routes on equal footing.

        Raises
        ------
        RuntimeError
            If `cond(full_hess)` of the boosted sibling model still exceeds
            `1e8` (this project's own numerical-safety line)
            -- i.e. if `regularization_boost` wasn't large enough. Do not
            silently return a number in that case; a larger
            `regularization_boost` is needed, or the dual route is not safe
            here at all (see Step 6's `K_kappa`-scale caveat below).
        """
        X_train, y_train_col = self.training_points[None][0]
        y_train = y_train_col.ravel()
        n = len(y_train)

        boosted_kwargs = dict(self._init_kwargs)
        boosted_kwargs["regularization_weight"] = regularization_boost
        boosted_kwargs.setdefault("print_global", False)
        boosted = RMTB(**boosted_kwargs)
        boosted.set_training_values(X_train, y_train)
        boosted.train()

        H = boosted.full_hess.toarray()
        cond_H = np.linalg.cond(H)
        if cond_H > 1e8:
            raise RuntimeError(
                f"dual route unsafe at regularization_boost={regularization_boost:.0e}: "
                f"cond(full_hess)={cond_H:.3e} exceeds the 1e8 safety line. "
                "Try a larger regularization_boost."
            )

        Phi, _, _c = boosted.full_jac_dict[0]
        Phi = Phi.toarray()
        Hinv = np.linalg.inv(H)
        PhiHinv = Phi @ Hinv
        K_kappa = PhiHinv @ Phi.T
        # NOTE: a safe cond(full_hess) does not by itself guarantee this route
        # is accurate -- the kappa_xx self-term below can still suffer
        # catastrophic cancellation if K_kappa's own scale is very large
        # relative to 1 (Step 6's finding, Q20's second addendum). Not
        # checked automatically here; cross-check against
        # `predict_variances()` on the same query points before trusting
        # this route on a new dataset.
        chol_dual = cho_factor(K_kappa + np.eye(n), lower=True)

        F = self._basis_row(x)
        Kself = Hinv @ F.T
        kappa_xx = np.sum(F * Kself.T, axis=1)
        kstar_train = PhiHinv @ F.T
        tmp = cho_solve(chol_dual, kstar_train)
        var = self.sigma2 * (kappa_xx - np.sum(kstar_train * tmp, axis=0))
        return np.clip(var, 0.0, None)

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Predict mean/uncertainty, matching the interface shared by
        `smoothing_splines_uq.csaps_uq.NdGridCubicSmoothingSplineUQ`.

        Parameters
        ----------
        x_pred : array-like of shape (M, D)
        return_std, return_ci : bool
        alpha : float
            Significance level for the 95%-style intervals (default 0.05).

        Returns
        -------
        np.ndarray (just the mean) if neither flag is set, else a dict with
        'mean', 'var', 'std', and -- if return_ci -- 'ci_lower'/'ci_upper'
        (uncertainty of the mean surface) and 'pi_lower'/'pi_upper'
        (uncertainty of a new noisy observation, i.e. variance + sigma^2).
        Always uses `predict_variances()` (the weight-space route) -- call
        `predict_variances_dual()` directly for the cross-check route.
        """
        x_pred = np.asarray(x_pred, float)
        mean = self.predict_values(x_pred).ravel()
        if not (return_std or return_ci):
            return mean

        var = self.predict_variances(x_pred)
        std = np.sqrt(var)
        res = {"mean": mean, "var": var, "std": std}
        if return_ci:
            z = norm.ppf(1.0 - alpha / 2.0)
            res["ci_lower"] = mean - z * std
            res["ci_upper"] = mean + z * std
            res["pi_lower"] = mean - z * np.sqrt(var + self.sigma2)
            res["pi_upper"] = mean + z * np.sqrt(var + self.sigma2)
        return res
