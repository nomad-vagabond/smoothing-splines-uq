"""Shared dataset generation, model fitting, and timing helpers for the
speed-benchmark scripts in this package.

Importing this module does not time or fit anything by itself; every
function here is called explicitly by the `testN_*.py` scripts, so each
script stays independently runnable and reproducible (same seeds, same
synthetic target function) without depending on another script having run
first.

A note on thread pinning: this module does *not* set any BLAS/OpenMP
thread-count environment variables itself, because by the time Python
reaches this import, `smoothing_splines_uq`'s own top-level `__init__.py`
has already imported numpy, which is too late for those variables to take
effect. Instead, every `testN_*.py` entry-point script sets them as the
very first lines in the file, before importing anything from numpy or
this package -- see the top of any `testN_*.py` module for the exact
four-line pattern.
"""
import contextlib
import io
import time
from pathlib import Path

import numpy as np

from smoothing_splines_uq.polyharmonic import PolyharmonicSmoothingSpline
from smoothing_splines_uq.rmtb_uq import RMTBUQ
from smoothing_splines_uq.csaps_uq import NdGridCubicSmoothingSplineUQ
from smt.surrogate_models import KRG

RESULTS_DIR = Path(__file__).parent / "results"

SIGMA2_TRUE = 0.05 ** 2
XLIMITS = np.array([[0., 1.], [0., 1.]])
N_QUERY = 2000


def true_f(X):
    return np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1])


def make_grid_dataset(n_side, seed):
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, n_side)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    X = np.column_stack([GX.ravel(), GY.ravel()])
    Z = true_f(X).reshape(n_side, n_side) + np.sqrt(SIGMA2_TRUE) * rng.standard_normal((n_side, n_side))
    y = Z.ravel()
    return dict(kind="grid", X=X, y=y, g=g, Z=Z)


def make_scattered_dataset(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 2))
    y = true_f(X) + np.sqrt(SIGMA2_TRUE) * rng.standard_normal(n)
    return dict(kind="scattered", X=X, y=y)


def standard_datasets():
    """The six datasets (three grid sizes, three scattered sizes) used
    throughout the main sweep (Tests 1-2). Same seeds every call, so
    every script that calls this gets byte-identical data."""
    return {
        "small grid (n=64)": make_grid_dataset(8, seed=0),
        "medium grid (n=256)": make_grid_dataset(16, seed=1),
        "big grid (n=625)": make_grid_dataset(25, seed=2),
        "small scattered (n=30)": make_scattered_dataset(30, seed=10),
        "medium scattered (n=250)": make_scattered_dataset(250, seed=11),
        "big scattered (n=600)": make_scattered_dataset(600, seed=12),
    }


def shared_query_points(n_query=N_QUERY, seed=99):
    return np.random.default_rng(seed).uniform(0, 1, size=(n_query, 2))


def time_call(fn, reps=5):
    """Minimum wall-clock time over `reps` repeated calls, after one
    untimed warm-up call -- standard micro-benchmark practice, robust to
    one-off scheduling noise."""
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def quiet(fn):
    """SMT's KRG prints internal diagnostic noise during its MLE search --
    not an error, just verbose; silenced so benchmark stdout stays
    readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn()


def fit_rmtbuq(X, y, regularization_weight=1e-14):
    m = RMTBUQ(xlimits=XLIMITS, order=4, approx_order=2, num_ctrl_pts=20,
               energy_weight=1e-2, regularization_weight=regularization_weight,
               solver="dense-lu", print_global=False)
    m.set_training_values(X, y)
    m.train()
    return m


def fit_krg(X, y):
    krg = quiet(lambda: KRG(corr="matern32", poly="linear", print_global=False, n_start=3))
    krg.set_training_values(X, y)
    quiet(krg.train)
    return krg


def fit_tps(X, y):
    return PolyharmonicSmoothingSpline(X, y, m=2, lam=1e-3, sigma2=SIGMA2_TRUE)


def fit_csaps(g, Z, off_grid_method="exact"):
    return NdGridCubicSmoothingSplineUQ([g, g], Z, smooth=0.95, off_grid_method=off_grid_method)


# Name -> callable(model, Xq) -> variance array. Shared by every script
# that needs to time or call prediction for one of these four models
# (used directly by test6_headline_comparison.py; test3_variance_predict_time.py
# uses the identical VARIANCE_FNS dict below instead, so that script's own
# import list is self-descriptive).
#
# All four of these are now genuinely variance-ONLY calls, confirmed
# directly from source, not assumed: predict_variances()/.variance()
# compute nothing else. NdGridCubicSmoothingSplineUQ's own .variance()
# used to accept only grid-tuple queries, which made it unusable for the
# scattered Xq used throughout this benchmark suite -- the headline
# comparison previously had to fall back to a combined
# predict(..., return_std=True) call, producing a mean+variance-combined
# number that wasn't comparable on equal footing to the other three rows.
# .variance() was extended (see csaps_uq.py) to also accept scattered
# (N, ndim) query points, exactly mirroring the dual convention
# predict()/predict_mean() already used, so this asymmetry no longer
# exists.
PREDICT_FNS = {
    "KRG": lambda model, Xq: model.predict_variances(Xq).ravel(),
    "PolyharmonicSmoothingSpline": lambda model, Xq: model.variance(Xq),
    "RMTBUQ": lambda model, Xq: model.predict_variances(Xq),
    "NdGridCubicSmoothingSplineUQ": lambda model, Xq: model.variance(Xq),
}

# Mean-only and variance-only dispatch, for all four models -- each pair
# below is a genuinely separable call (confirmed from source: neither
# computes the other quantity as a side effect). NdGridCubicSmoothingSplineUQ's
# predict_mean()/variance() were added specifically to make this split
# possible here; previously there was no public API on that class that
# computed just one of mean/variance at scattered query points. Used by
# test2_mean_predict_time.py (MEAN_FNS) and test3_variance_predict_time.py
# (VARIANCE_FNS) respectively.
MEAN_FNS = {
    "KRG": lambda model, Xq: model.predict_values(Xq).ravel(),
    "PolyharmonicSmoothingSpline": lambda model, Xq: model.mean(Xq),
    "RMTBUQ": lambda model, Xq: model.predict_values(Xq).ravel(),
    "NdGridCubicSmoothingSplineUQ": lambda model, Xq: model.predict_mean(Xq),
}

VARIANCE_FNS = {
    "KRG": lambda model, Xq: model.predict_variances(Xq).ravel(),
    "PolyharmonicSmoothingSpline": lambda model, Xq: model.variance(Xq),
    "RMTBUQ": lambda model, Xq: model.predict_variances(Xq),
    "NdGridCubicSmoothingSplineUQ": lambda model, Xq: model.variance(Xq),
}


def time_predictions(fitted_by_dataset, Xq, fn_dict, reps=5, shuffle_seed=0):
    """Times `fn_dict[model_name](model, Xq)` for every (dataset, model)
    pair in `fitted_by_dataset` (a dict of dataset_name -> {model_name:
    model, ...}, as produced by calling `fit_all_models` once per dataset
    *before* calling this function), and returns a list of result dicts.

    Why this function exists, not just a loop: fitting KRG at a large `n`
    takes tens of seconds of sustained, heavy computation, and a Python
    process's effective CPU clock speed (turbo/boost state) can still be
    ramping up or settling for a short while after a burst like that ends.
    If fitting and timing are interleaved per dataset -- fit KRG, fit the
    others, then immediately time predictions, move to the next dataset --
    whatever CPU-state effect that heavy KRG fit leaves behind lands
    disproportionately on the *next* measurement, which also happens to be
    the one at the next (larger) `n`. That confound was confirmed directly
    in this project's own investigation: timing RMTBUQ's prediction alone,
    with no other model's fit immediately beforehand, is flat versus `n`
    in both ascending and descending order; interleaved with the other
    models' fits as in the loop this function replaces, it was not.

    The fix is to finish *all* fitting (all models, all datasets) before
    timing *anything*, and to time the resulting (dataset, model) pairs in
    an order that is deliberately uncorrelated with `n` -- shuffled with a
    fixed seed, so runs stay reproducible. By the time timing starts here,
    every dataset's heavy fits have already happened, so any leftover
    CPU-state effect applies roughly uniformly rather than tracking `n`.
    """
    pairs = []
    for dataset_name, fitted in fitted_by_dataset.items():
        for model_name, entry in fitted.items():
            # `fitted` is `fit_all_models`'s own return format,
            # {model_name: (model, fit_time_seconds)} -- unwrap the tuple.
            model = entry[0] if isinstance(entry, tuple) else entry
            if model_name in fn_dict:
                pairs.append((dataset_name, model_name, model))

    order = list(range(len(pairs)))
    import random
    random.Random(shuffle_seed).shuffle(order)

    rows = []
    for idx in order:
        dataset_name, model_name, model = pairs[idx]
        fn = fn_dict[model_name]
        t = time_call(lambda model=model, fn=fn: fn(model, Xq), reps=reps)
        rows.append({"dataset": dataset_name, "model": model_name, "time_ms": t * 1000})
    return rows


def fit_all_models(dataset):
    """Fits KRG, PolyharmonicSmoothingSpline, and RMTBUQ on every dataset;
    additionally fits NdGridCubicSmoothingSplineUQ when the dataset is a
    complete grid (it cannot be fit on scattered data at all).

    Returns {model_name: (fitted_model, fit_time_seconds)}.
    """
    X, y = dataset["X"], dataset["y"]
    fitted = {}

    t0 = time.perf_counter()
    krg = fit_krg(X, y)
    fitted["KRG"] = (krg, time.perf_counter() - t0)

    t0 = time.perf_counter()
    tps = fit_tps(X, y)
    fitted["PolyharmonicSmoothingSpline"] = (tps, time.perf_counter() - t0)

    t0 = time.perf_counter()
    rmtb = fit_rmtbuq(X, y)
    fitted["RMTBUQ"] = (rmtb, time.perf_counter() - t0)

    if dataset["kind"] == "grid":
        t0 = time.perf_counter()
        csaps_m = fit_csaps(dataset["g"], dataset["Z"], off_grid_method="exact")
        fitted["NdGridCubicSmoothingSplineUQ"] = (csaps_m, time.perf_counter() - t0)

    return fitted
