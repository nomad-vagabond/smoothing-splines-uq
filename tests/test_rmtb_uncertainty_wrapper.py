"""`RMTBUncertaintyWrapper` wired to a real, trained `RMTBUQ` model.

This is a code-correctness check, not a math/methodology check -- the two
classes compute the *same* already-validated formula
($\\mathrm{Var}_{\\mathrm{bayes}}=\\sigma^2f^\\top\\mathrm{mtx}^{-1}f$,
`04_rmtb_variance_weight_route.ipynb` Section 1.2/Test 1) through two
separately-written pieces of code. Traced down to LAPACK, both ultimately
call the same Cholesky factor-and-solve (`?POTRF`/`?POTRS`), so agreement
here is not an independent numerical or derivation-level confirmation of
the formula (that's Test 2 and Notebook 5's job respectively) -- it only
catches implementation bugs (a sign flip, a transpose, an off-by-one in
`edf`) in `RMTBUncertaintyWrapper`'s generic `fit`/`predict`. That is
exactly why it belongs here, as a unit test, rather than as a standing
notebook cell.
"""
import numpy as np

from smoothing_splines_uq.rmtb_uq import RMTBUQ
from smoothing_splines_uq.validation.rmtb_uncertainty_wrapper import RMTBUncertaintyWrapper

RMTB_KW = dict(xlimits=np.array([[0., 1.], [0., 1.]]), order=4, approx_order=2,
               num_ctrl_pts=20, energy_weight=1e-2, regularization_weight=1e-14,
               solver="dense-lu", print_global=False)


def _make_dataset(spacing, n_side=8, seed=1):
    rng = np.random.default_rng(seed)
    if spacing == "regular":
        g = np.linspace(0, 1, n_side)
        G1, G2 = np.meshgrid(g, g, indexing="ij")
        X = np.column_stack([G1.ravel(), G2.ravel()])
    else:
        X = rng.uniform(0, 1, size=(n_side * n_side, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + 0.05 * rng.standard_normal(len(X))
    return X, y


def _fit_pair(spacing):
    X, y = _make_dataset(spacing)
    m = RMTBUQ(**RMTB_KW)
    m.set_training_values(X, y)
    m.train()

    Phi, _PhiT, _c = m.full_jac_dict[0]
    Phi = Phi.toarray()
    H = m.full_hess.toarray()

    wrapper = RMTBUncertaintyWrapper(ridge_alpha=0.0)
    wrapper.fit(Phi, y, H)
    return m, wrapper


def test_wrapper_weight_vector_matches_rmtbuq_solution():
    for spacing in ("regular", "random"):
        m, wrapper = _fit_pair(spacing)
        assert np.max(np.abs(wrapper.c_hat - m.sol.ravel())) < 1e-8


def test_wrapper_sigma2_matches_rmtbuq():
    for spacing in ("regular", "random"):
        m, wrapper = _fit_pair(spacing)
        assert abs(wrapper.sigma2 - m.sigma2) < 1e-10


def test_wrapper_variance_matches_rmtbuq_on_dense_grid():
    gq = np.linspace(0, 1, 60)
    GQ1, GQ2 = np.meshgrid(gq, gq, indexing="ij")
    Xq_dense = np.column_stack([GQ1.ravel(), GQ2.ravel()])

    for spacing in ("regular", "random"):
        m, wrapper = _fit_pair(spacing)
        F_dense = m._basis_row(Xq_dense)
        var_rmtbuq = m.predict_variances(Xq_dense)
        var_wrapper = wrapper.predict(F_dense)["epistemic_var"]
        gap = np.max(np.abs(var_rmtbuq - var_wrapper))
        assert gap / np.ptp(var_rmtbuq) < 1e-8


def test_wrapper_sigma2_diverges_from_rmtbuq_near_exact_interpolation():
    """Documents a known discrepancy, not a bug to fix here.

    `RMTBUQ`'s own Wahba estimator floors its denominator at
    `max(n-edf, 1e-8)`; `RMTBUncertaintyWrapper`'s floors it at
    `max(1.0, n-edf)`. Identical whenever `n-edf` is safely above `1`
    (true in every other test in this file, and in every synthetic case
    this project used before a real, scattered dataset pushed `edf`
    genuinely close to `n` -- `07_rans_real_world_validation.ipynb`
    Test 1). A very small `energy_weight` reproduces that same
    near-exact-interpolation regime here, synthetically: `c_hat` and the
    dense-grid variance still agree (the matrix algebra is identical),
    but `sigma2` itself diverges by orders of magnitude, exactly tracking
    the two floors' ratio -- not a sign of a bug in either class, but a
    convention difference a future change to either floor should keep in
    mind.
    """
    X, y = _make_dataset("random")
    n = len(X)

    kw = dict(RMTB_KW)
    kw["energy_weight"] = 1e-12
    kw["regularization_weight"] = 1e-8  # keeps cond(mtx) safe enough for the wrapper's own guard
    m = RMTBUQ(**kw)
    m.set_training_values(X, y)
    m.train()
    assert n - m.edf < 1e-4  # confirm this really is the near-exact-interpolation regime

    Phi, _PhiT, _c = m.full_jac_dict[0]
    Phi = Phi.toarray()
    H = m.full_hess.toarray()
    wrapper = RMTBUncertaintyWrapper(ridge_alpha=0.0)
    wrapper.fit(Phi, y, H)

    assert np.max(np.abs(wrapper.c_hat - m.sol.ravel())) < 1e-6  # same weight vector throughout

    dof_rmtbuq = max(n - m.edf, 1e-8)
    dof_wrapper = max(1.0, n - m.edf)
    expected_ratio = dof_wrapper / dof_rmtbuq
    assert expected_ratio > 100  # the two floors actually differ substantially here
    observed_ratio = m.sigma2 / wrapper.sigma2
    assert abs(observed_ratio - expected_ratio) / expected_ratio < 1e-6
