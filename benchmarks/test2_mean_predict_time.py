"""Test 2 -- mean (surface value) prediction time vs. n.

What this tests: how each model's own mean-only call (the
interpolated/smoothed surface value, with no uncertainty) scales with
`n`, against the structural claim that `RMTBUQ`'s cost is set by its
basis resolution rather than by training-set size. Same shared
`N_QUERY=2000` query points for every model/dataset combination.

Methodology note (fit-then-time, not fit-and-time-per-dataset): all
models for all datasets are fit first, and only then is prediction
timed, in an order shuffled independently of `n` (`common.time_predictions`).
See `test3_variance_predict_time.py` and `common.time_predictions` for
why -- the same fix applies here, with its own independent shuffle seed.

Usage: python -m smoothing_splines_uq.benchmarks.test2_mean_predict_time
Output: results/test2_mean_predict_time.csv (columns: dataset, kind, n, model, predict_time_ms)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

from smoothing_splines_uq.benchmarks import common


def main():
    datasets = common.standard_datasets()
    Xq = common.shared_query_points()

    print("Fitting all models on all datasets first (no timing yet)...")
    fitted_by_dataset = {name: common.fit_all_models(d) for name, d in datasets.items()}

    print("Timing MEAN-only predictions, in an order shuffled independently of n...")
    timed_rows = common.time_predictions(fitted_by_dataset, Xq, common.MEAN_FNS, shuffle_seed=1)

    kind_by_dataset = {name: d["kind"] for name, d in datasets.items()}
    n_by_dataset = {name: len(d["X"]) for name, d in datasets.items()}

    rows = []
    for r in timed_rows:
        rows.append({
            "dataset": r["dataset"],
            "kind": kind_by_dataset[r["dataset"]],
            "n": n_by_dataset[r["dataset"]],
            "model": r["model"],
            "predict_time_ms": r["time_ms"],
        })
        print(f"{r['dataset']:28s} {r['model']:30s} {r['time_ms']:10.3f} ms")

    df = pd.DataFrame(rows)
    dataset_order = list(datasets.keys())
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = common.RESULTS_DIR / "test2_mean_predict_time.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
