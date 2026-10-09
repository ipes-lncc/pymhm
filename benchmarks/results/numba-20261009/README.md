# Compiled array kernels: a focused CPU comparison

These records compare the original PyMHM array implementation at revision
`55359af7220ad9db9d9255e6eae30a7beb243789` with the source digests recorded in
`compiled-2d.json` and `compiled-3d.json`. Both use the same locked dependencies,
acquisition helper, physical data, discretization and one native BLAS thread.
The host is an Intel Xeon Silver 4216 workstation; affinity and software versions
are included in each report. Measurements use separate project timing windows,
without an exclusive operating-system allocation.

The unit square/cube has exact pressure given by the product of `sin(pi*x_i)`.
Scalar permeability is `exp(0.25*prod(sin(2*pi*x_i/0.137)))`. The source is the
independently differentiated `-div(K*grad(p))`, checked against centered
differences in `source-control.json`. Local pressure is continuous P2; each
macroface has a constant P0 normal-flux trace and each macroelement retains its
physical-volume pressure moment. Homogeneous pressure is imposed weakly at the
boundary. No equivalent-material matrix or factor reuse is enabled.

| Case | Macro simplices | Local simplices in total | Original warm median | Compiled warm median | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D: macro 4, local refinement 8 | 32 | 2,048 | 4.408 s | 4.180 s | 1.054× |
| 3D: macro 2, local refinement 4 | 48 | 3,072 | 13.644 s | 12.604 s | 1.083× |

The timer includes fresh geometry and problem descriptions, native thread setup,
local assembly/factorization, ordered global assembly, solution/reconstruction,
analytical field norms, original local-equation controls, macro balance and
pressure samples. Three fresh workflows follow the first workflow; their median
reuses compiled code, while rebuilding operators and factors. Each process
starts with an absent Numba cache directory. First-call times include JIT and
ordinary native initialization. Imports and interpreter launch/exit have
separate receipts. The 2D warm ranges overlap; its small median difference does
not establish a general simulation speedup. This is a serial CPU comparison,
without MPI, GPU or strong/weak scalability measurements.

The P0 skeleton is deliberately coarse: pressure/physical Darcy-flux L2 errors
are approximately 0.0363/0.5116 in 2D and 0.1180/0.9397 in 3D. This comparison
establishes implementation equivalence, rather than high accuracy or an
asymptotic convergence rate. Maximum pressure-coefficient differences are
8.77e-15 and 5.88e-15. Original local L2 backward errors, global compatibility
and macro skeleton balance keep their unchanged 1e-10 limits. Macro balance
uses the conservative skeleton flux; fine-cell conservation is not asserted
for the raw primal gradient flux.

`comparison.json` contains measured ratios and report digests. The four raw JSON
reports retain every complete/phase duration, physical check, deterministic
microbenchmark definition, version and executed source digest. Their NPZ files
archive pressure/trace coefficients, CSC structure and RHS, actual fine meshes,
and the executed local nodal basis matrices with individual digests. Paired
geometry, basis and CSC structures are identical. Field replay must retain each
recorded basis matrix and its orientation convention.

`micro-equivalence.json` checks deterministic tensor Gram inputs and 1,024 SPD
25×25 response blocks sharing 128 trace coordinates. Their timings omit
tabulation, callbacks and solvers; the component ratios are separate from
complete workflow ratios. Actual MHM response reduction is timed separately.
`revision-control.json` verifies the original source snapshot against its Git
tree. `launch-receipts.json` records launch/import scopes. `figure-provenance.json`
links each publication figure to its measured report and plotting helper.

Use the package-only, step-by-step
[Darcy notebook](../../../notebooks/darcy/numba_kernel_performance.ipynb) for
explicit equations, assembly, reconstruction and field plots. Its teaching
illustration enriches the traces and states its own discretization. For repeat
acquisition, use the importable
[measurement helper](../../../examples/numba_performance.py) in the locked `test`
environment:

```python
from pathlib import Path
from examples.numba_performance import KernelCase, acquire

acquire((KernelCase(2, 4, 8),), Path("build/numba-repeat/compiled-2d.json"), label="compiled-2d")
```

Run each dimension in a fresh process, set `NUMBA_CACHE_DIR` to a new empty
directory, and set BLAS/OpenMP/Numba thread counts to one before importing the
package. Use a separate original-source checkout/snapshot with the same helper
and locked environment for the baseline; verify its imported module paths and
source hashes before comparing reports. Record outer process/import costs
separately. `compare_reports` verifies coefficient archive digests and paired
fields. `plot_comparison` in
[the plotting helper](../../../examples/plot_numba_performance.py) regenerates
the PNG/SVG/PDF figures. Figures are attributed to the IPES Research Group under
CC BY 4.0; the numerical records follow the repository data policy.
