# Windows

PyMHM's portable core targets native Windows x86-64. It uses the same
`Equation`, `LocalEquations`, `MultiscaleProblem` and `LocalProblem` contracts
as Linux and macOS. The checked-in Pixi lockfile includes Windows NumPy, SciPy,
Basix and optional Intel MKL/PyPardiso packages. PETSc is optional.

The Windows CI jobs are configured to collect core coverage and exercise native
PARDISO, signed shared-face assembly, local physical moments and serial/thread/process reconstruction.
They also install a built wheel outside the checkout and repeat the generic
local/global checks against an independently assembled conforming system.
An environment resolved in the lockfile or a Linux test run does not establish
that a particular revision passed native Windows execution; inspect that
revision's [Tests workflow](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml).
Validated release tags automatically publish the portable package and its
documentation; [development](development.md) describes the required release settings.

## Installation

For an application, install the portable package from PyPI with Python 3.11–3.13.
For example, with Python 3.12 installed, run these commands in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pymhm
.\.venv\Scripts\python.exe -c "import pymhm; print(pymhm.__version__)"
```

For Intel MKL PARDISO, install `"pymhm[intel]"` instead of `pymhm` in the same
environment. These commands use the virtual environment's interpreter directly;
activation, a repository checkout and Pixi are unnecessary. See the
[installation guide](installation.md) for extras and upgrades.
The pip installation provides the coefficient/Basix core. Native UFL assembly
requires a separate DOLFINx environment; see
[native UFL assembly](installation.md#native-ufl-assembly) for the supported
Linux/macOS setup.

### Repository development with Pixi

Install [Pixi for Windows](https://pixi.prefix.dev/latest/installation/) version 0.76.2 and
open PowerShell in a checkout. These commands use the existing lockfile:

```powershell
pixi --version
pixi list --locked --no-install -e test-core
pixi install --locked -e test-core
pixi run --locked -e test-core pytest -q tests/test_windows_portability.py
pixi run --locked -e test-core coverage-run
```

Coverage qualification combines Linux core and native FEM measurements from
the same revision and checks both 99% thresholds. The
[coverage procedure](development.md#coverage) describes those commands.

For Intel MKL PARDISO:

```powershell
pixi install --locked -e intel
pixi run --locked -e intel python -c "import pypardiso"
pixi run --locked -e intel pytest -q tests/test_windows_portability.py
```

Use `pixi run --locked -e notebooks jupyter lab` for portable notebooks.
The coefficient-based notebooks in `notebooks/foundations/operators` introduce
the generic API. Cells calling the DOLFINx adapter require a supported native
FEM environment; the `introduction` FEM environment is not a native Windows profile.
See the [notebook catalogue](tutorials.md) for the physical examples and methods.

Pip commands resolve their own environment. Use Pixi's locked profiles
for the repository's reproducible development and scientific checks.

## Local assembly and solver selection

Windows uses the same user-defined local and global equations. A provider
returns `LocalEquations` or assembled `LocalProblem` blocks; numerical
condensation and shared-face reduction remain in PyMHM. Arrays, sparse matrices,
external simulation packages and learned local responses use that common
contract. Basix remains the reference-element provider.

| Capability | Native Windows profile |
| --- | --- |
| Generic local/global coefficient DSL, retained modes and physical moments | Portable core |
| Basix reference elements and built-in portable assembly | Portable core |
| Sparse direct LU | `scipy` (SuperLU), or optional `pypardiso` |
| Krylov methods | `cg`, `minres`, `gmres`, with their stated operator conditions |
| CPU algebraic multigrid | Optional `pyamg`; available in the locked test/notebook profiles |
| Ordered serial, thread and spawn-process local execution | Portable core |
| Native DOLFINx/PETSc adapter and PETSc/MPI workflows | Current FEM/HPC profiles exclude Windows |
| CUDA solvers | Separate optional GPU profile; CPU portability does not qualify Windows GPU execution |

Choose local and global solvers explicitly with `SolverConfig` as explained in
the [variational guide](variational.md). For example,
`SolverConfig(local_solver="pypardiso", global_solver="scipy")` keeps the
global solver on SciPy and selects PARDISO for each local factorization.
`pypardiso-symmetric` selects real symmetric-indefinite factorization;
symmetry and the original matrix residual are checked. The PyPardiso wrapper
accepts real matrices and right-hand sides; use SciPy for complex systems.
PyPardiso's upstream package supports [Linux and Windows](https://github.com/haasad/PyPardiso).

For an isolated virtual environment that inherits the locked profile's Python
dependencies, PyPardiso also needs that profile's MKL runtime. The installed-wheel
check binds `PYPARDISO_MKL_RT` to the active library and records its SHA-256.

Use the default double-precision refinement mode. An explicit extended mode
requires a NumPy `longdouble` type wider than float64, which standard Windows
builds do not provide. PyMHM checks that requirement and reports an unsupported
request without changing the numerical tolerances.

Scientific acquisition commands that expose `--refinement-precision` can select
`double` explicitly. Their records identify the executed mode and coefficient
precision; replay validates that contract. Scientific defaults requiring extended
precision still require a platform that actually provides it.

Source and pinned text inputs use LF line endings in Git checkouts, and acquisition
manifests use POSIX relative paths on every platform. Byte digests remain literal:
changing an input or an archived field invalidates its recorded identity.

## Spawn workers and notebooks

Process execution uses `spawn` on every operating system. Put worker callables
in an importable Python helper and protect executable entry points with
`if __name__ == "__main__":`. In a notebook, import that helper before selecting
`ExecutionConfig(backend="process", workers=8, native_threads=1)`; a callable
defined only in an interactive cell is not an importable worker entry point.
The notebook still declares the formulation step by step, while the helper
provides the importable callback required by Python's process runtime.

Every worker constructs and releases its native solver resources. Transfer
ordinary coefficient arrays and metadata rather than live factors or native
FEM/CUDA objects. The coordinator reduces shared-face contributions in declared
cell order. Explicit process counts above 61 are rejected on Windows; the
default respects that Python executor limit. Native thread limits and worker
counts should fit the available CPUs. With several PARDISO workers, begin with
one MKL thread per process and measure the complete workflow.

PyMHM serializes PARDISO construction, factorization, solve and cleanup calls
inside each process. Each prepared factor retains its own lifetime. Spawned
processes have independent locks and factors, so local problems can execute in
parallel across processes; MKL can also use the declared native threads inside
each factorization. This host-call policy follows the
[PyPardiso 0.4.7 wrapper's concurrency requirements](https://github.com/haasad/PyPardiso/blob/v0.4.7/pypardiso/scipy_aliases.py).

## Native FEM scope

The current PyMHM DOLFINx integration uses the locked Unix FEM stack and PETSc
objects. Its native Windows execution is not qualified by the portable-core
or PARDISO tests. Use the Linux profiles in WSL2 for those notebooks and adapters.

The current [DOLFINx installation guidance](https://github.com/FEniCS/dolfinx/blob/main/README.md)
also describes native Windows conda packages in beta testing, without PETSc or
`petsc4py`, and a Visual Studio requirement for just-in-time compilation. That
upstream route is distinct from PyMHM's current adapter and is not advertised
as a verified PyMHM native Windows FEM backend.
