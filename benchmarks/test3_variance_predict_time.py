"""Test 3 -- variance prediction time vs. n.

What this tests: the structural claim this project's `RMTBUQ` work is
built around -- that its query cost is set by basis resolution, not
training-set size -- against three genuine kernel methods (KRG,
PolyharmonicSmoothingSpline, and, on grid data, NdGridCubicSmoothingSplineUQ)
whose query cost is expected to grow with `n`. Same shared N_QUERY=2000
query points for every model/dataset combination.

Methodology note (fit-then-time, not fit-and-time-per-dataset): all
models for all datasets are fit first, and only then is prediction
timed, in an order shuffled independently of `n` (`common.time_predictions`).
An earlier version of this script fit and timed each dataset in
sequence -- fit KRG, fit the others, time prediction, move to the next
(larger) dataset -- which let whatever CPU-state effect a large, slow
KRG fit leaves behind land disproportionately on the very next
measurement, at the next (larger) `n`. That produced an apparent
*decrease* in RMTBUQ's own prediction time with growing `n`, which is
not real: timed in isolation, with no other model's fit immediately
before it, RMTBUQ's prediction time is flat regardless of `n`, in both
ascending and descending dataset order (confirmed directly as part of
this project's own investigation into the earlier, misleading result).
See `common.time_predictions` for the full explanation.

All four models expose a genuinely variance-only call, timed here via
`common.VARIANCE_FNS` (`predict_variances()`/`.variance()` for KRG,
PolyharmonicSmoothingSpline, and RMTBUQ; `NdGridCubicSmoothingSplineUQ`'s
own `.variance()` was extended to also accept scattered off-grid query
points, see `csaps_uq.py`, so this is variance-only for it too, not a
mean+variance combined cost).

Usage: python -m smoothing_splines_uq.benchmarks.test3_variance_predict_time
Output: results/test3_variance_predict_time.csv (columns: dataset, kind, n, model, predict_time_ms)
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

    print("Timing VARIANCE-only predictions, in an order shuffled independently of n...")
    timed_rows = common.time_predictions(fitted_by_dataset, Xq, common.VARIANCE_FNS, shuffle_seed=2)

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
    out_path = common.RESULTS_DIR / "test3_variance_predict_time.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
