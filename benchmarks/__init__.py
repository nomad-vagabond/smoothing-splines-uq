"""Terminal-run speed benchmarks for Notebook 8 (RMTBUQ vs. Kriging).

Each `testN_*.py` module in this package is a standalone script: run it
directly (`python -m smoothing_splines_uq.benchmarks.test1_fit_time`) and it
fits models, times them, and writes its results to a CSV file under
`results/`. Nothing here runs anything at import time, and nothing in this
package uses threads, multiprocessing, or any other concurrency -- every
script measures one model on one dataset at a time, strictly in sequence,
so wall-clock comparisons between models are not confounded by whatever
else the machine happens to be doing concurrently.

Notebook 8 does not run any of these benchmarks itself; it only loads the
CSV files these scripts produce (see `results/`) and analyzes/plots them.
Re-running a script overwrites its own output file(s); nothing here is
destructive to anything outside this package's own `results/` directory.

`run_all.py` runs all five scripts back to back, each in its own
subprocess, one at a time.
"""
