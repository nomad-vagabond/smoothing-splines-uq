"""Test 5 -- do the engineering choices behind these classes pay off?

Part 1: `RMTBUQ.predict_variances()` factors `mtx` once and solves every
query in one batched Cholesky call. Checked directly against a
deliberately naive reimplementation of the same formula (same factored
`mtx`, solved one query at a time in a Python loop), on the big scattered
dataset.

Part 2: `NdGridCubicSmoothingSplineUQ`'s `off_grid_method="fast"` builds a
`RectBivariateSpline` re-interpolant at construction time, trading a small
off-grid accuracy cost (quantified separately in Notebook 2) for speed.
Checked here on both grid datasets: how much speed does that trade
actually buy?

Usage: python -m smoothing_splines_uq.benchmarks.test5_engineering_choices
Output:
    results/test5_batched_vs_naive.csv     (one row: batched_ms, naive_ms,
                                             speedup, max_diff)
    results/test5_csaps_exact_vs_fast.csv  (dataset, exact_ms, fast_ms,
                                             speedup, max_diff)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from smoothing_splines_uq.benchmarks import common

BATCHED_VS_NAIVE_DATASET = "big scattered (n=600)"
CSAPS_DATASETS = ["small grid (n=64)", "big grid (n=625)"]


def batched_vs_naive(datasets, Xq):
    d = datasets[BATCHED_VS_NAIVE_DATASET]
    m = common.fit_rmtbuq(d["X"], d["y"])
    mtx = m.mtx.toarray()
    chol = cho_factor(mtx, lower=True)

    def var_naive_loop(Xq_):
        F = m._basis_row(Xq_)
        out = np.empty(len(Xq_))
        for i, f0 in enumerate(F):
            out[i] = m.sigma2 * (f0 @ cho_solve(chol, f0))
        return out

    t_batched = common.time_call(lambda: m.predict_variances(Xq))
    t_naive = common.time_call(lambda: var_naive_loop(Xq), reps=3)

    var_batched = m.predict_variances(Xq)
    var_naive = var_naive_loop(Xq)

    row = {
        "dataset": BATCHED_VS_NAIVE_DATASET,
        "n": len(d["X"]),
        "batched_ms": t_batched * 1000,
        "naive_loop_ms": t_naive * 1000,
        "speedup": t_naive / t_batched,
        "max_diff": np.max(np.abs(var_batched - var_naive)),
    }
    print(f"batched={row['batched_ms']:.2f}ms  naive_loop={row['naive_loop_ms']:.2f}ms  "
          f"speedup={row['speedup']:.1f}x  max_diff={row['max_diff']:.3e}")
    return pd.DataFrame([row])


def csaps_exact_vs_fast(datasets, Xq):
    rows = []
    for name in CSAPS_DATASETS:
        d = datasets[name]
        m_exact = common.fit_csaps(d["g"], d["Z"], off_grid_method="exact")
        m_fast = common.fit_csaps(d["g"], d["Z"], off_grid_method="fast")

        def call_exact(m=m_exact):
            return m.predict(Xq, return_std=True)["var"]

        def call_fast(m=m_fast):
            return m.predict(Xq, return_std=True)["var"]

        t_exact = common.time_call(call_exact)
        t_fast = common.time_call(call_fast)
        var_exact = call_exact()
        var_fast = call_fast()

        rows.append({
            "dataset": name,
            "n": len(d["X"]),
            "exact_ms": t_exact * 1000,
            "fast_ms": t_fast * 1000,
            "speedup_exact_over_fast": t_exact / t_fast,
            "max_diff": np.max(np.abs(var_exact - var_fast)),
        })
        print(f"{name:20s} exact={t_exact*1000:8.3f}ms  fast={t_fast*1000:8.3f}ms  "
              f"speedup={t_exact/t_fast:6.1f}x  max_diff={np.max(np.abs(var_exact - var_fast)):.3e}")
    return pd.DataFrame(rows)


def main():
    datasets = common.standard_datasets()
    Xq = common.shared_query_points()

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    naive_df = batched_vs_naive(datasets, Xq)
    naive_path = common.RESULTS_DIR / "test5_batched_vs_naive.csv"
    naive_df.to_csv(naive_path, index=False)
    print(f"Wrote {len(naive_df)} rows to {naive_path}")

    csaps_df = csaps_exact_vs_fast(datasets, Xq)
    csaps_path = common.RESULTS_DIR / "test5_csaps_exact_vs_fast.csv"
    csaps_df.to_csv(csaps_path, index=False)
    print(f"Wrote {len(csaps_df)} rows to {csaps_path}")


if __name__ == "__main__":
    main()
