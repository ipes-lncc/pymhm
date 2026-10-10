# Choosing linear solvers and reusing factors

Configure local and global algebra independently on your existing problem:

```python
from pymhm import SolverConfig, assemble

system = assemble(
    problem,
    solvers=SolverConfig(local_solver="scipy", global_solver="scipy"),
)
solution = system.solve()
```

A local mean-constrained Neumann operator and the global skeleton system are
generally indefinite. Choose a solver that accepts their actual matrix structure;
an elliptic AMG preset is intended for an admissible positive-definite block or
projected complement. The table below distinguishes those capabilities.

## Select a backend

`solve_linear(A, b, solver=...)` accepts a finite square dense or sparse matrix
and a vector or multiple right-hand sides. Inputs are normalized to float64 or
complex128. Every returned column must satisfy

$$
\lVert Ax-b\rVert_2\leq\max(\mathrm{atol},\mathrm{rtol}\lVert b\rVert_2).
$$

This residual check is independent of the backend's stopping report. It does not
bound forward error for an ill-conditioned operator. Singular direct solves,
iteration failures, nonfinite values and incompatible dimensions raise errors.
Requested optional solvers never silently fall back to another backend.

UFL/DOLFINx assembly and numerical solver selection are independent. The
native FEM adapters return SciPy/NumPy operators without importing PETSc;
`SolverConfig(local_solver="scipy", global_solver="scipy")` uses the same
portable solvers as coefficient-defined problems. Optional solver choices
retain the restrictions below. MPI runtime requirements for DOLFINx meshes
remain even when SciPy or PARDISO performs the solve; distributed PETSc assembly
is a separate capability.

A small residual alone does not establish uniqueness: an unstable trace/local
space combination can have a compatible right-hand side and an almost singular
skeleton. SciPy LU therefore equilibrates rows and columns explicitly and checks
that every pivot exceeds `eps * n * max(abs(U))`. This is a numerical rank
diagnostic, not proof of exact singularity. Severely ill-conditioned systems may
also be rejected. `validate_invertible(A)` exposes the same sparse diagnostic for
other backends, and the global hybrid solve applies it before an optional backend
solve. Highly unequal physical scales are balanced before the check; the final
residual is still evaluated in the original equations.

| Backend | Method | Restrictions |
|---|---|---|
| `scipy` | SuperLU direct LU | Portable real and complex baseline. |
| `cg` | SciPy conjugate gradients | Hermitian positive definite; symmetry is checked. |
| `minres` | SciPy MINRES | Real symmetric; indefinite operators are permitted. |
| `gmres` | SciPy GMRES | Nonsymmetric operators are permitted. |
| `pypardiso` | Intel MKL PARDISO | Optional native runtime; this wrapper supports real matrices/RHS only. |
| `pypardiso-symmetric` | Intel MKL PARDISO, symmetric indefinite | Explicit `mtype=-2`; checked real symmetry and full-original-matrix residuals. Does not require positive definiteness. |
| `pypardiso-symmetric-matching` | Intel MKL PARDISO, symmetric indefinite | Explicit weighted matching, scaling and 1×1/2×2 pivot profile; original residual checks. |
| `petsc` | PETSc/MUMPS pivoted LU | Requires PETSc built with MUMPS; uses `COMM_SELF`. Complex data requires a complex PETSc build. |
| `petsc-symmetric` | PETSc/MUMPS pivoted symmetric-indefinite LDLt | Checked real symmetry; explicitly disables the SPD assumption. Uses `COMM_SELF` and checks the full original equations. |
| `cupy` | CUDA sparse QR | Optional CuPy/CUDA, includes host/device transfers. |
| `cudss` | NVIDIA cuDSS through nvmath | Optional CuPy/nvmath/cuDSS; reusable device factorization. |
| `pyamg` | CPU smoothed aggregation + CG | Positive-definite elliptic operators; optional near-nullspace candidates. |
| `amgx` | GPU aggregation AMG + FGMRES | Real positive-definite elliptic operators; separately installed AmgX/pyamgx. |

Positive diagonals and symmetry are checked for the two AMG presets, but these
checks do not prove positive definiteness. Their intended application is an
elliptic block or a constrained positive-definite local problem. The full MHM
skeleton/kernel system is generally indefinite. Its solution requires an
appropriate saddle-point solver or block preconditioner; passing it directly to
an elliptic AMG preset is not a supported shortcut.

## Native library initialization and worker limits

`preload_solver_backend(name)` loads only the explicitly selected backend's
libraries. It constructs no explicit numerical factors, AMG hierarchy,
communicator or GPU session. PETSc preparation imports its Python package;
PETSc/MPI initialization remains in the existing numerical adapter.
Call it before a new `threadpool_limits` context when managing native thread
counts directly, so that every loaded BLAS/OpenMP pool receives that policy.

The ordered local executors prepare their named solver before operator assembly
and before applying `native_threads`. Providers, compilers and custom solvers can
supply an idempotent `prepare_runtime()` hook for the same library initialization
step. Numerical resources remain local to their worker and are released through
`close()` after execution. The hook is optional; portable callables need no
additional protocol. Preparation does not change precision, native parameter
profiles or the residual criterion.
Library preparation leaves operator capability checks in their numerical owners.
An unavailable optional library is reported when an admissible operator actually
requires that backend; unused local spaces need no native solver initialization.

PARDISO native construction, factorization, solve and cleanup entries are
serialized within each process. Separate factors retain independent lifetimes;
a single factor must not be used concurrently. Independent spawned processes
have independent native locks. Linux native tests exercise serial, thread and
spawn execution. Windows uses the same portable execution and cleanup contracts;
its native runtime qualification is reported by Windows CI separately.

## Reusing direct factorizations

```python
import numpy as np
from scipy import sparse
from pymhm.linalg.linear import factorize

A = sparse.csr_matrix([[3.0, 1.0], [1.0, 2.0]])
with factorize(A, solver="scipy") as factors:
    first = factors.solve(np.array([1.0, 2.0]))
    second = factors.solve(np.eye(2))
```

`factorize` supports SciPy, PARDISO, PETSc and cuDSS. The matrix is copied and
factorized once. Repeated RHS solves reuse the factorization. The context manager
releases resources on both success and failure. A factorization is mutable backend
state and must not be used concurrently by several threads. CuPy sparse QR and
AMG/Krylov methods are exposed through `solve_linear`, not represented as LU
factorizations.

`pypardiso-symmetric` explicitly selects symmetric-indefinite factorization for
real saddle matrices. It supplies the upper triangle with all diagonal entries,
including structural zeros in multiplier rows. Symmetry is checked relative to
the matrix scale. This profile retains MKL's automatic parameter defaults.
`pypardiso-symmetric-matching` selects an explicit profile with scaling, weighted
matching, 1×1/2×2 symmetric pivots, METIS ordering, matrix checks and up to five
native refinement steps, with a pivot-perturbation threshold of \(10^{-13}\).
The profiles are separate choices: matching is not uniformly more accurate,
and neither is an automatic fallback for the other. Residuals and correction
equations use the complete original matrix and the requested tolerance.

`petsc-symmetric` requests the MUMPS symmetric-indefinite factorization through
PETSc's Cholesky interface, with `MAT_SPD=False`. This selects pivoted LDLt;
it does not assume that a saddle matrix is positive definite. The distinction
follows the [PETSc MUMPS adapter](https://petsc.org/release/src/mat/impls/aij/mpi/mumps/impl/imumps.c.html).
The ordinary `petsc` backend retains its LU factorization.

Reusable direct factors also accept `equilibration="symmetric"`. Five [Ruiz (2001)](https://epubs.stfc.ac.uk/manifestation/271/raltr-2001034.pdf)
infinity-row sweeps construct a positive diagonal \(D\), and the backend solves
\(DADy=Db\), followed by \(x=Dy\). This explicit congruence preserves symmetry
and the equations; it adds no regularization. Symmetry and nonzero rows are
required. The final residual and all correction equations are checked against
\(A\) and \(b\) in their original units. The default is `equilibration="none"`;
the option is independent of the selected backend's internal scaling. It is also
available through `solve_linear` for reusable direct backends.

Direct factors also accept
`factors.solve(rhs, refinement_precision="extended")`. The factorization and
correction solves remain in their backend's double precision; accumulated
corrections and the returned solution use a wider NumPy long-double type.
By default, at most two corrections are attempted, with the same original
residual criterion. The mode is explicit and unavailable on platforms where long double
has no additional precision. It does not change pivot diagnostics, tolerances
or the selected backend. For the bound variational workflow, select
`SolverConfig(local_refinement_precision="extended")`; local lifts and
reconstructed fields retain their additional digits. Local AMG
projection does not support this option. The same explicit mode is available
through `solve_linear(..., refinement_precision="extended")` for both Krylov
backends and reusable direct factors (SciPy, PARDISO, PETSc and cuDSS).

The explicit `refinement_steps` argument controls the maximum number of outer
defect corrections in `factors.solve` and `solve_linear`. It accepts a
nonnegative integer and defaults to two. Zero checks only the initial backend
solution; a larger limit reuses the same factorization or Krylov preconditioner.
For example, `factors.solve(rhs, refinement_precision="extended",
refinement_steps=8)` permits eight corrections while preserving the requested
residual criterion. Acceptance can occur earlier; exhaustion, stagnation or
divergence still raises `LinearSolveError`. This limit is separate from the
backend's internal refinement settings. AmgX reuses its prepared GPU hierarchy
for these original-equation corrections. CuPy sparse QR exposes no outer
correction loop and rejects nondefault limits.

`HybridSystem.solve(refinement_precision="extended")` independently enables
extended accumulation for the global solve and reconstructed coefficients.
Its optional `rtol` sets the global algebraic tolerance explicitly. With `rtol=None`,
a new factor uses the default tolerance and an existing prepared factor preserves
its own tolerance; an explicit conflicting tolerance is rejected. Extended
accumulation alone does not force corrections after the global residual has
already passed. Local physical balances must still meet their separate checks.
Local and global refinement precision are independent:
`SolverConfig(global_refinement_precision="extended")` selects wider global
accumulation in the bound workflow. Wider accumulation does not make the
factorization exact or guarantee accuracy for every condition number.

`refine_hybrid(system, solution, ...)` optionally corrects the complete original
local and weak-trace equations after condensation. It supports distinct test
and trial spaces, several retained modes, prescribed trace coefficients and
physical mean constraints. Supply the same `boundary_load`, `fixed` data and
`moments=[(local_weights, target), ...]` used in the original problem. The defect
loads are solved with the original local operators and harmonic lifts; the
method and approximation spaces are unchanged. Local factors are released
cellwise, and the full uncondensed matrix is never assembled.

Acceptance uses the norm of the original local rows, free weak-trace rows and
supplied physical moments. Its right-hand-side norm includes prescribed-trace
elimination in the same input units. This algebraic quantity is separate from
finite-element error norms. `HybridRefinement.residual_norms` records the initial
residual and each correction; `solution` holds the resulting fields. Persist
these fields together with their evaluation basis. Reconstructing only from the
corrected trace/coarse coefficients and the original lifts omits the additional
defect-source responses. The returned `solution.residual` evaluates the original
free weak-trace and retained-test rows on the corrected fields; it excludes
artificial gauge rows. Gauge multipliers accumulate all corrections. `max_steps=0`
checks the initial state only. Exhaustion raises `LinearSolveError` with the
achieved residual and threshold. The default hybrid solve does not invoke this
additional work. Explicit `refinement_precision="extended"` retains correction
digits and has the same platform requirement as extended direct refinement.

For prescribed trace values, the final physical-equation check includes a
componentwise bound on rounding in their elimination. In row \(i\), with
\(m_i\) prescribed nonzero entries, the bound is
\(\gamma_{m_i+1}(|A_{FD}|\,|x_D|)_i\), where
\(\gamma_n=n\epsilon/(1-n\epsilon)\). This accounts for cancellation of
balanced Neumann data without assigning a relative tolerance to each large
boundary contribution. Incompatible physical loads above this rounding bound
remain errors, including when the resulting net load is small.

The PETSc adapter explicitly selects MUMPS. Local kernel constraints, mixed
formulations and the global pressure gauge introduce saddle blocks with zero or
structurally absent diagonal entries. PETSc's internal sequential LU is not a
general replacement for a pivoted factorization of these matrices. The adapter
checks that PETSc was built with MUMPS and raises an actionable error if it was
not; it does not shift the diagonal or substitute another backend. This package
does not expose PETSc options that change this preset. See the official
[PETSc MUMPS interface](https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/).

The cuDSS preset enables maximum diagonal-product matching and five iterative
refinement steps for zero-diagonal saddle blocks. Every solution must satisfy
the same residual criterion as the other backends.

For local multiscale basis construction, one factorization solves the source
and all trace right-hand sides together. A GPU factorization can therefore
amortize setup over several local basis vectors, but transfer and setup costs
remain significant for small local problems. See the measured
[performance cases](https://github.com/ipes-lncc/pymhm/blob/main/docs/performance.md).

## CPU algebraic multigrid

```python
import numpy as np
import pyamg
from pymhm.linalg.linear import solve_linear

A = pyamg.gallery.poisson((32, 32), format="csr")
b = np.ones(A.shape[0])
x = solve_linear(
    A,
    b,
    solver="pyamg",
    near_nullspace=np.ones((A.shape[0], 1)),
)
```

The candidate columns are the slowly varying modes that the hierarchy should
represent on coarse levels. Scalar diffusion often uses constants; elasticity
needs appropriate rigid-motion candidates after essential constraints have been
applied. These are near-nullspace candidates, not automatic removal of an exact
singular kernel. MHM mean constraints must still be imposed consistently.

The preset constructs one smoothed-aggregation hierarchy for all supplied RHS
columns and applies its V-cycle as a SciPy CG preconditioner. It uses local Jacobi
prolongator weighting to avoid the random spectral-radius estimate used by the
default diagonal-weighted prolongator. This is a fixed, inspectable preset;
problem-dependent AMG tuning and block saddle preconditioners remain separate
work. See [PyAMG's aggregation API](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html).

## GPU algebraic multigrid

`solver="amgx"` uses NVIDIA AmgX through pyamgx with device-double mode `dDDI`,
FGMRES, one aggregation V-cycle per preconditioner application, block-Jacobi
smoothing and a dense coarse solve. Matrix/RHS transfer, hierarchy setup and
resource destruction are included in the call. One hierarchy is reused across
RHS columns and independently checked true-residual corrections. The original
operator and each original RHS define the residual criterion, including zero
forcing. Relative-tolerance solves normalize every nonzero column by its largest
absolute entry and undo this scale in the solution. This preserves the equations
while keeping small physical loads above the native convergence test's
[absolute stopping floor](https://github.com/NVIDIA/AMGX/blob/91a8413ef267b1c32aff4014c02820e1c5897ac2/src/convergence/relative_ini.cu).
The original residual acceptance criterion is unchanged.

For repeated loads on one fixed elliptic operator, prepare the hierarchy explicitly:

```python
from pymhm.linalg.linear import prepare_amgx

with prepare_amgx(A, rtol=1e-10) as prepared:
    solution = prepared.solve(rhs_columns)
    another_solution = prepared.solve(another_rhs, refinement_steps=4)
```

The hierarchy belongs to that copied operator. A different material matrix needs
a different setup. It accepts real symmetric operators with positive diagonals;
the caller must establish positive definiteness. Closing the context releases
all native vectors, solver, matrix, resource and configuration handles.

For MHM Neumann locals, the shared condensation owner projects compatibility,
pins one independent coordinate per declared kernel mode and restores the
declared physical moment constraints after each correction. AmgX receives the
pinned elliptic operator. Source, oriented trace and finite-precision retained
mode responses share one hierarchy across every projected correction pass.
Compatibility pairings, source moments and gauge restoration use the same
extended or compensated accumulator as the original residual checks; native
corrections and response storage remain double precision. Every permitted
correction, including the fourth, is checked before acceptance.
Their projected original-equation target is `1e-12`; each pinned solve keeps its
`1e-10` criterion. Reconstructed fields also require an independent check of
the original physical equations; the reduced global residual is insufficient.
Trial and test kernels/moments must match, and a general coarse basis is outside
this local AMG interface.

AmgX and pyamgx have separate native build requirements. They are not silently
installed by the standard Python package extra. Follow the
[NVIDIA build instructions](https://github.com/NVIDIA/AMGX#building) and
[pyamgx installation guide](https://pyamgx.readthedocs.io/en/latest/install.html).
A working CUDA runtime and matching native library are required.

The adapter owns AmgX's process-global initialization/finalization and serializes
its own resource lifecycles. Do not overlap it with an independently managed
pyamgx session in the same process. Each prepared hierarchy uses one GPU.
Several independent local problems can be assigned to separate spawned GPU
processes, each setting `CUDA_VISIBLE_DEVICES` before its first CUDA import.
Within one process the lifecycle lock remains held until the prepared hierarchy
is closed. The adapter uses the stated fixed preset and does not accept external
AmgX configuration dictionaries.

## Combine solver choice with execution

Pass `SolverConfig(local_solver=..., global_solver=...)` to `assemble(problem)`.
The [CPU guide](guides/cpu.md) configures serial, thread or spawned work;
[MPI](guides/mpi.md) distributes the global algebra, and
[CUDA](guides/gpu.md) assigns local matrices to devices.
For unchanged operators and changing loads, use
[offline preparation](execution.md#repeated-sources-and-boundary-values).
Solver selection does not change quadrature, approximation spaces or material.

## AMG block preconditioners for mixed saddle systems

`pymhm.linalg.block.SaddleBlockSolver` accepts a real symmetric matrix partitioned as

$$
\mathcal A=\begin{bmatrix}A&B\\B^T&-C\end{bmatrix}.
$$

The `split` argument identifies the contiguous first block. GMRES acts on the
**original** matrix. Its block-diagonal preconditioner uses

$$
P=A+\alpha BB^T,\qquad S=C+B^T\operatorname{diag}(P)^{-1}B.
$$

Both preconditioner blocks must be numerically positive definite. The explicit
augmentation changes only the preconditioner. In particular, it can control the
kernel of a semidefinite first block through the coupling. The automatic positive
scale balances the largest entry of `A` against the largest diagonal of `BB.T`;
`augmentation=` may specify a different positive scale. Unsupported block
structures are rejected. This is not a general AMG solver for indefinite inputs.

```python
from pymhm.linalg.block import SaddleBlockSolver

with SaddleBlockSolver(system.matrix, system.trace_size, backend="pyamg") as block:
    solution = system.solve(factorization=block.as_factorization())
```

The example applies directly to an unconstrained condensed Darcy system whose
skeleton/retained ordering has that block form. For eliminated boundary DOFs or
additional gauges, prepare the actual constrained matrix and its appropriate
block split; the exact-matrix factorization contract rejects a mismatched matrix.

`backend="pyamg"` retains a CPU smoothed-aggregation hierarchy. `backend="amgx"`
retains an NVIDIA AmgX aggregation hierarchy on the GPU. Each preconditioner
application performs one V-cycle on `P`; `S` uses a reusable sparse LU. Outer
Krylov operations, the Schur block and numerical checks remain on the CPU.
The GPU path uploads/downloads block vectors at every application. Native GPU
resources belong to the solver context and are released on exit.

For several source columns, reuse the same prepared block solver through its
context manager. Its factorization must match the actual constrained matrix;
selecting a block split from an unconstrained layout after boundary elimination
is invalid. Original-equation residual checks apply to every returned column.

Performance measurements and physical applications are in the
[Gallery](gallery/index.md). This guide specifies solver capabilities and
configuration rather than selecting a fastest backend for all problem sizes.
