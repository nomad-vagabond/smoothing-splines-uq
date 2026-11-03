# smoothing-splines-uq

Bayesian posterior variance for tensor-product smoothing splines, as a
strictly additive layer on top of two production curve-fitting libraries
that otherwise ship mean-only: SMT's `RMTB` and `csaps`'s
`NdGridCubicSmoothingSpline`.

## Why this exists

Gaussian process regression (Kriging) gives you a posterior mean *and* a
posterior variance for free, but costs `O(n^3)` to fit and `O(n)` per query
for `n` training points. Tensor-product smoothing splines -- SMT's `RMTB`
on scattered data, `csaps`'s grid splines on regular grids -- fit a fixed
basis of size `m << n` once, so every later query costs `O(m)`, independent
of how much data trained the model. The catch: as shipped, neither library
computes a posterior variance at all.

This package closes that gap, using the classical Kimeldorf--Wahba
correspondence (1970): a penalized smoothing spline is -- exactly, not
approximately -- the posterior mean of a specific Gaussian process. That
identity gives a closed-form Bayesian posterior covariance over a spline's
own fitted coefficients, derivable two independent ways (a weight-space
route operating on the `m x m` basis, and a dual, kernel-space route
operating on the `n x n` data Gram matrix), which this package implements,
cross-checks against each other, and attaches to `RMTB` and `csaps` as
thin, additive subclasses -- never touching how either library computes
its own mean.

See the accompanying manuscript (cited below) for the full derivation,
numerical validation across synthetic benchmarks and a real
Reynolds-Averaged Navier--Stokes aerodynamic dataset, and discussion of
where this does and doesn't match an isotropic Kriging reference exactly.

## Status

Research code backing an in-progress manuscript. The public API
(`__init__.py`'s `__all__`) is exercised by the test suite and the eight
validation notebooks below; everything else (`validation/`,
`benchmarks/`) is a supporting tool, not a stability guarantee. There is
no `pyproject.toml`/`setup.py` yet, so it isn't pip-installable -- see
[Installation](#installation).

## Installation

Clone the repository and put its parent directory on your `PYTHONPATH` (or
drop the `smoothing_splines_uq/` folder into a project that already has
these dependencies installed):

```bash
git clone git@github.com:nomad-vagabond/smoothing-splines-uq.git
pip install numpy scipy csaps smt matplotlib
```

```python
import sys
sys.path.insert(0, "/path/to/parent/of/smoothing_splines_uq")
import smoothing_splines_uq as ssuq
```

Running the test suite or notebooks works the same way: put the
directory containing your clone of this repository on `PYTHONPATH`.

## Quick start

### `RMTBUQ` -- scattered data, SMT's `RMTB` with posterior variance

```python
import numpy as np
from smoothing_splines_uq import RMTBUQ

X = np.random.default_rng(0).uniform(0, 1, size=(60, 2))
y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + 0.05 * np.random.randn(60)

model = RMTBUQ(xlimits=np.array([[0., 1.], [0., 1.]]),
                num_ctrl_pts=20, energy_weight=1e-2, print_global=False)
model.set_training_values(X, y)
model.train()

Xq = np.random.default_rng(1).uniform(0, 1, size=(500, 2))
out = model.predict(Xq, return_std=True, return_ci=True)  # dict: mean, std, var, ci_lower, ci_upper, pi_lower, pi_upper
variance_only = model.predict_variances(Xq)                # skip the mean if you only need variance
variance_dual = model.predict_variances_dual(Xq)            # independent cross-check route (slower; validation use)
```

### `NdGridCubicSmoothingSplineUQ` -- regular grids, `csaps` with posterior variance

```python
import numpy as np
from smoothing_splines_uq import NdGridCubicSmoothingSplineUQ

x1 = np.linspace(0, 1, 15)
x2 = np.linspace(0, 1, 15)
Y = np.sin(3 * x1[:, None]) * np.cos(3 * x2[None, :])

model = NdGridCubicSmoothingSplineUQ([x1, x2], Y, smooth=0.95)

mean_only = model.predict_mean([x1, x2])        # on-grid query, exact
std_only = np.sqrt(model.variance([x1, x2]))
scattered = model.predict(np.random.default_rng(0).uniform(0, 1, size=(200, 2)),
                           return_std=True)       # off-grid, off_grid_method="exact" by default
```

### `PolyharmonicSmoothingSpline` -- exact, closed-form isotropic reference

An exact Duchon/polyharmonic smoothing spline, used throughout this
project's own validation as a numerically-exact baseline (no finite basis,
no grid) that neither `RMTB`'s finite-rank basis nor `csaps`'s anisotropic
boundary behavior is expected to match exactly:

```python
from smoothing_splines_uq import PolyharmonicSmoothingSpline

spline = PolyharmonicSmoothingSpline(X, y, m=2, lam="reml", sigma2="reml")
mean = spline.mean(Xq)
var = spline.variance(Xq)
```

## Package layout

| Module | Contents |
|---|---|
| `uk.py` | Foundational solvers (`saddle_point_solve`/`predict`, `gls_predict`, `saddle_point_variance`) everything else is built on. |
| `kernels.py` | Reproducing kernels (Wiener/IWP2, Matérn 1/2--5/2) used by the lower-level solvers and wrappers. |
| `polyharmonic.py` | `PolyharmonicSmoothingSpline` -- the general N-D Duchon spline with calibrated variance. |
| `smoothing_spline.py` | `SmoothingSpline1D`/`SmoothingSpline2D` -- thin variance companions for `scipy`'s own smoothing splines. |
| `csaps_uq.py` | `NdGridCubicSmoothingSplineUQ` -- grid-exact tensor-product splines, subclassing `csaps`. |
| `rmtb_uq.py` | `RMTBUQ` -- scattered-data energy-regularized B-splines, subclassing SMT's `RMTB`. |
| `plotting.py` | 2-D map and 1-D CI/PI band plotting helpers shared by the validation notebooks. |
| `validation/` | Independent re-implementations used only to cross-check the main classes above; not part of the modeling API. |

## Validation notebooks

Eight notebooks, in increasing realism, covering the three hypotheses this
package's math rests on (exact mean equivalence; closed-form weight-space
and dual posterior variance; Kriging-like spatial behavior):

1. `01_1d_identity_kriging_vs_splines.ipynb` -- exact 1-D mean/variance
   identity against Kriging, plus a six-implementation agreement check.
2. `02_csaps_mean_and_variance.ipynb` -- 2-D grid variance for `csaps`.
3. `03_rmtb_vs_rmtc_conditioning.ipynb` -- numerical conditioning, `RMTB`
   vs. SMT's `RMTC`.
4. `04_rmtb_variance_weight_route.ipynb` -- weight-space variance behavior.
5. `05_rmtb_H_singularity_and_dual_route.ipynb` -- the energy matrix's null
   space and the dual/kernel-space route.
6. `06_rmtb_kriging_like_uq_behavior.ipynb` -- spatial texture, dimple
   formation, and `RMTB` vs. Kriging under tuned hyperparameters.
7. `07_rans_real_world_validation.ipynb` -- a real 35-point RANS
   aerodynamic wing dataset.
8. `08_speed_benchmark_rmtb_vs_kriging.ipynb` -- query-speed benchmark vs.
   Kriging, confirming `O(m)` cost independent of training-set size.

## Tests

```bash
PYTHONPATH=<path to the directory containing your clone> pytest smoothing_splines_uq/tests/
```

## Benchmarks

Wall-clock fit/query-time benchmarks, each run in its own subprocess so
BLAS/OpenMP thread pinning takes effect cleanly:

```bash
python -m smoothing_splines_uq.benchmarks.run_all
```

Results are written as CSVs under `benchmarks/results/`.

## Citation

If you use this package, please cite the accompanying manuscript:

> Morlier, J. and Pasko, V. *Bayesian Uncertainty Quantification for
> Tensor-Product Smoothing Splines: Mathematical Foundations, Dual
> Formulations, and Numerical Validation.* Manuscript in preparation.

## License

Not yet specified -- add a `LICENSE` file before relying on this code
outside this project.
