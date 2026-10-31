"""Runs all five speed-benchmark scripts back to back, each in its own
subprocess, strictly one after another -- never concurrently. Each
subprocess gets a completely fresh Python process, so the BLAS
thread-pinning every `testN_*.py` script sets at its own top takes effect
cleanly every time, and nothing a prior script left in memory can affect
the next one's timings.

Safe to re-run: each script only ever overwrites its own CSV file(s)
under `results/`.

Usage: python -m smoothing_splines_uq.benchmarks.run_all
"""
import subprocess
import sys

SCRIPTS = [
    "test1_fit_time",
    "test2_mean_predict_time",
    "test3_variance_predict_time",
    "test4_weight_vs_dual",
    "test5_engineering_choices",
]


def main():
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run([sys.executable, "-m", f"smoothing_splines_uq.benchmarks.{name}"])
        if result.returncode != 0:
            print(f"\n!!! {name} failed (exit code {result.returncode}); stopping.", file=sys.stderr)
            sys.exit(result.returncode)
    print("\nAll benchmark scripts completed.")


if __name__ == "__main__":
    main()
