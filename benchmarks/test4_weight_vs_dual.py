"""Test 4 -- RMTBUQ weight route vs. dual route prediction speed.

Both routes compute the same posterior variance (Notebook 5); this asks
which is faster to *query*, on two scattered datasets, using the
`regularization_weight=1e-6` boost Notebook 5 already validated as the
mitigation for the dual route's own near-singular `full_hess`.

A second part diagnoses *why* the dual route is slower than the formula
alone would suggest: a line-by-line replica of `predict_variances_dual()`,
timed with cumulative checkpoints within one execution (self-consistent
by construction, since each checkpoint is a running total within a single
call rather than three separately-timed calls on a possibly
differently-loaded machine), isolating how much of the cost is the
avoidable refit-and-invert of an internal sibling model versus the
unavoidable per-query solve.

Usage: python -m smoothing_splines_uq.benchmarks.test4_weight_vs_dual
Output:
    results/test4_weight_vs_dual.csv      (dataset, n, cond_full_hess, safe,
                                            weight_route_ms, dual_route_ms, ratio)
    results/test4_dual_diagnostic.csv     (one row, at n=600: refit_ms,
                                            refit_invert_cumulative_ms, total_ms,
                                            invert_share, max_diff_vs_shipped)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from smt.surrogate_models import RMTB

from smoothing_splines_uq.benchmarks import common

DUAL_DATASETS = ["small scattered (n=30)", "big scattered (n=600)"]
DIAGNOSTIC_DATASET = "big scattered (n=600)"


def weight_vs_dual_comparison(datasets, Xq):
    rows = []
    for name in DUAL_DATASETS:
        d = datasets[name]
        m_boosted = common.fit_rmtbuq(d["X"], d["y"], regularization_weight=1e-6)
        cond_h = np.linalg.cond(m_boosted.full_hess.toarray())

        t_weight = common.time_call(lambda m=m_boosted: m.predict_variances(Xq))
        t_dual = common.time_call(lambda m=m_boosted: m.predict_variances_dual(Xq, regularization_boost=1e-6))

        rows.append({
            "dataset": name,
            "n": len(d["X"]),
            "cond_full_hess": cond_h,
            "safe_lt_1e8": cond_h < 1e8,
            "weight_route_ms": t_weight * 1000,
            "dual_route_ms": t_dual * 1000,
            "ratio_dual_over_weight": t_dual / t_weight,
        })
        print(f"{name:28s} weight={t_weight*1000:8.2f}ms  dual={t_dual*1000:8.2f}ms  "
              f"ratio={t_dual/t_weight:6.2f}x  cond(full_hess)={cond_h:.3e}")
    return pd.DataFrame(rows)


def dual_route_diagnostic(datasets, Xq):
    d = datasets[DIAGNOSTIC_DATASET]
    m_diag = common.fit_rmtbuq(d["X"], d["y"], regularization_weight=1e-6)

    def dual_replica_with_checkpoints():
        t_start = time.perf_counter()

        boosted_kwargs = dict(m_diag._init_kwargs)
        boosted_kwargs["regularization_weight"] = 1e-6
        boosted_kwargs.setdefault("print_global", False)
        boosted = RMTB(**boosted_kwargs)
        boosted.set_training_values(d["X"], d["y"])
        boosted.train()
        t_refit = time.perf_counter() - t_start

        H = boosted.full_hess.toarray()
        Hinv = np.linalg.inv(H)
        t_refit_invert = time.perf_counter() - t_start

        Phi, _, _c = boosted.full_jac_dict[0]
        Phi = Phi.toarray()
        PhiHinv = Phi @ Hinv
        K_kappa = PhiHinv @ Phi.T
        chol_dual = cho_factor(K_kappa + np.eye(len(d["X"])), lower=True)

        F = m_diag._basis_row(Xq)
        Kself = Hinv @ F.T
        kappa_xx = np.sum(F * Kself.T, axis=1)
        kstar_train = PhiHinv @ F.T
        tmp = cho_solve(chol_dual, kstar_train)
        var = m_diag.sigma2 * (kappa_xx - np.sum(kstar_train * tmp, axis=0))
        t_total = time.perf_counter() - t_start

        return t_refit, t_refit_invert, t_total, var

    dual_replica_with_checkpoints()  # one untimed warm-up call
    breakdowns = [dual_replica_with_checkpoints() for _ in range(6)]
    t_refit, t_refit_invert, t_total, var_replica = min(breakdowns, key=lambda b: b[2])

    var_shipped = m_diag.predict_variances_dual(Xq, regularization_boost=1e-6)

    row = {
        "dataset": DIAGNOSTIC_DATASET,
        "n": len(d["X"]),
        "refit_ms": t_refit * 1000,
        "refit_invert_cumulative_ms": t_refit_invert * 1000,
        "total_ms": t_total * 1000,
        "invert_share": (t_refit_invert - t_refit) / t_total,
        "refit_share": t_refit / t_total,
        "remaining_share": (t_total - t_refit_invert) / t_total,
        "max_diff_vs_shipped": np.max(np.abs(var_replica - var_shipped)),
    }
    print(f"\nDual-route diagnostic ({DIAGNOSTIC_DATASET}): "
          f"refit={row['refit_ms']:.1f}ms ({row['refit_share']:.0%}), "
          f"+invert={row['refit_invert_cumulative_ms']:.1f}ms ({row['invert_share']:.0%}), "
          f"total={row['total_ms']:.1f}ms, "
          f"max|replica-shipped|={row['max_diff_vs_shipped']:.3e}")
    return pd.DataFrame([row])


def main():
    datasets = common.standard_datasets()
    Xq = common.shared_query_points()

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = weight_vs_dual_comparison(datasets, Xq)
    comparison_path = common.RESULTS_DIR / "test4_weight_vs_dual.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Wrote {len(comparison_df)} rows to {comparison_path}")

    diagnostic_df = dual_route_diagnostic(datasets, Xq)
    diagnostic_path = common.RESULTS_DIR / "test4_dual_diagnostic.csv"
    diagnostic_df.to_csv(diagnostic_path, index=False)
    print(f"Wrote {len(diagnostic_df)} rows to {diagnostic_path}")


if __name__ == "__main__":
    main()
