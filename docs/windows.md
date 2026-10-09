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
documentation; [development](development/contributing.md) describes the required release settings.

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
[native UFL assembly](installation.md#native-ufl-assembly) and the
[native FEM scope](#native-fem-scope) below for installation and qualification.

To open a downloaded notebook, install the notebook extra and launch JupyterLab
with the same interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install "pymhm[notebooks]"
.\.venv\Scripts\python.exe -m jupyterlab primal_galerkin.ipynb
```

The library installation contains no example helpers, notebooks or case data.
The downloaded notebook's first cell verifies and extracts its separate companion
ZIP, then imports the workspace's inspectable `examples/` and `scripts/` helpers.
Selected data and recorded figures use verified separate links in the
[data catalogue](data.md). No clone or Pixi installation is required.

The notebook uses `.pymhm-companions/<checksum>` below the current directory
as its writable workspace; `PYMHM_WORKSPACE` can select another directory. Outputs and acquired inputs are
written there. Native notebooks require the separately installed backend stated
in the catalogue; use a compatible DOLFINx Conda environment and install
`"pymhm[notebooks]"` into it for native UFL examples.

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
[coverage procedure](development/contributing.md#coverage) describes those commands.

For Intel MKL PARDISO:

```powershell
pixi install --locked -e intel
pixi run --locked -e intel python -c "import pypardiso"
pixi run --locked -e intel pytest -q tests/test_windows_portability.py
```

Use `pixi run --locked -e notebooks jupyter lab` for portable notebooks.
The coefficient-based notebooks in `notebooks/foundations/operators` introduce
the generic API. Cells calling the DOLFINx adapter require the native FEM stack
and a working JIT compiler. Notebook sections that explicitly select PETSc or
distributed PETSc reference solves require that separate runtime.
See the [notebook catalogue](tutorials/notebooks.md) for the physical examples and methods.

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
| Native UFL/DOLFINx local assembly | Locked `fem` profile; native CSR/vector assembly with a single-rank local mesh |
| PETSc/MUMPS solvers and distributed PETSc assembly | Separate optional capability; the locked Windows FEM stack does not supply PETSc |
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

The optional UFL/DOLFINx adapter assembles matrices and vectors with DOLFINx's
native interfaces and copies them to SciPy/NumPy. It does not import
`dolfinx.fem.petsc` or require `petsc4py`. DOLFINx 0.9 and 0.10 are supported
by the adapter; use the platform's resolution in the checked-in lockfile.
Both the generic form compiler and the local hybrid adapter require a
single-rank mesh communicator. Create local meshes with `MPI.COMM_SELF`,
including inside spawned workers. DOLFINx still requires a compatible MPI
runtime even when the numerical solve uses SciPy or PARDISO.

Choose the local and global solvers independently of assembly. The default
`SolverConfig(local_solver="scipy", global_solver="scipy")` uses SuperLU;
optional PARDISO and the other solvers retain their matrix and platform
requirements. A selected `petsc` backend requires PETSc/MUMPS and does not
fall back silently. Distributed PETSc assembly and the three-dimensional
introductory notebook's distributed reference solves require the Unix PETSc
stack; WSL2 can supply that environment on a Windows host.

For a checkout, prepare and test the native Windows FEM profile in PowerShell:

```powershell
pixi install --locked -e fem
pixi run --locked -e fem python -c "import dolfinx, ufl; from mpi4py import MPI; print(dolfinx.__version__, MPI.Get_library_version())"
pixi run --locked -e fem test-fem-portable
pixi run --locked -e fem fem-portable-check
```

The locked `fem-intel` profile combines DOLFINx/UFL with Intel MKL PARDISO
on Windows and Linux. Use `SolverConfig(local_solver="pypardiso", global_solver="scipy")`
for local PARDISO factors, or choose `global_solver="pypardiso"` separately
when the global matrix satisfies that backend's requirements:

```powershell
pixi install --locked -e fem-intel
pixi run --locked -e fem-intel test-fem-portable
pixi run --locked -e fem-intel fem-portable-pardiso-check
```

The required native checks fail when their selected backend is unavailable
and write `build/reports/fem-portable-scipy.json` or
`build/reports/fem-portable-pardiso.json`. The records identify the platform,
dependency and solver-package versions, blocked PETSc imports, solver selections
and analytical field/residual controls. `source_revision` identifies Git HEAD;
`executed_source_sha256` identifies the actual tracked `src/pymhm` file contents,
including working-tree changes. The recorded source snapshot must remain unchanged
throughout the control. DOLFINx PETSc capability flags are recorded separately
from the absent PETSc Python modules. CI artifacts also identify their workflow revision.

The [DOLFINx/UFL sparse-solver notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/dolfinx_sparse_solvers.ipynb)
uses the `introduction-intel` notebook profile to demonstrate independent
local/global solver choices, signed Darcy traces, pressure moments and spawn
workers. Execute it with
`pixi run --locked -e introduction-intel notebooks-run foundations/operators/dolfinx_sparse_solvers.ipynb`.
Its native Windows execution requires its own qualification receipt.

Install Visual Studio with its C/C++ compiler and Windows SDK, and use a
developer terminal that makes the compiler available to DOLFINx/FFCx JIT.
The [official DOLFINx installation guidance](https://github.com/FEniCS/dolfinx/blob/main/README.md#conda)
describes its Windows Conda packages as beta and requires Visual Studio;
those packages do not include PETSc or `petsc4py`.

The Tests workflow configures actual Windows FEM assembly and solve checks,
including a process that blocks PETSc imports. Check the reports for the
identified revision before claiming native execution. Portable-core tests,
PARDISO tests, Linux FEM execution and lockfile resolution are separate evidence.
