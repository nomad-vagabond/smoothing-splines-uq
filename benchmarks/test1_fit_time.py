"""Test 1 -- fit time vs. training-set size `n`.

What this tests: `KRG`'s own MLE hyperparameter search is expected to
scale badly with `n`. `RMTBUQ`'s fit cost is expected to be governed by
`num_ctrl_pts` (fixed at 20 throughout this benchmark suite), not `n`.

Usage: python -m smoothing_splines_uq.benchmarks.test1_fit_time
Output: results/test1_fit_time.csv (columns: dataset, kind, n, model, fit_time_ms)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

from smoothing_splines_uq.benchmarks import common


def main():
    datasets = common.standard_datasets()

    rows = []
    for name, d in datasets.items():
        fitted = common.fit_all_models(d)
        for model_name, (_model, t_fit) in fitted.items():
            rows.append({
                "dataset": name,
                "kind": d["kind"],
                "n": len(d["X"]),
                "model": model_name,
                "fit_time_ms": t_fit * 1000,
            })
            print(f"{name:28s} {model_name:30s} {t_fit * 1000:10.2f} ms")

    df = pd.DataFrame(rows)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = common.RESULTS_DIR / "test1_fit_time.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
