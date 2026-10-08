# Installation and environments

The core supports Python 3.11–3.13 and requires NumPy, SciPy, threadpoolctl and `fenics-basix>=0.9`.
Basix supplies the built-in polynomial bases without requiring DOLFINx, PETSc or MPI.
Optional native dependencies are loaded only when their functionality is called.

## Install from PyPI with pip

Use Python 3.11, 3.12 or 3.13. A package installation needs no repository checkout
or Pixi installation. Create a [virtual environment](https://docs.python.org/3/library/venv.html)
for your application, then install `pymhm` from PyPI.

On Linux or macOS, with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pymhm
python -c "import pymhm; print(pymhm.__version__)"
```

On Windows, with Python 3.12 installed, run in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pymhm
.\.venv\Scripts\python.exe -c "import pymhm; print(pymhm.__version__)"
```

Replace `3.12` with `3.11` or `3.13` to select another supported interpreter.
The Windows commands call the virtual environment's interpreter directly and
do not require activation. In the commands below, use that interpreter in place
of `python` if your environment is not activated.

The default installation includes the generic local/global problem API, Basix
reference elements, portable assembly, SciPy direct and iterative solvers, and
serial, thread and process execution. Pip installs NumPy, SciPy, threadpoolctl
and Basix automatically. See the [variational guide](variational.md) for defining
your equations and selecting solvers, and the [Windows guide](windows.md) for
native Windows capabilities.

To update an existing installation:

```bash
python -m pip install --upgrade pymhm
```

### Optional pip extras

Install extras for the capabilities your application uses. Each extra includes
the same `pymhm` package and adds the dependencies listed below.

| Command | Additional capability | Requirements |
| --- | --- | --- |
| `python -m pip install "pymhm[amg]"` | CPU algebraic multigrid with PyAMG | Supported NumPy/SciPy environment |
| `python -m pip install "pymhm[intel]"` | Intel MKL PARDISO through PyPardiso | Linux or Windows x86-64; the extra has no PyPardiso dependency on other platforms |
| `python -m pip install "pymhm[mpi]"` | MPI communication through mpi4py | Compatible MPI runtime and launcher; follow the [mpi4py installation guide](https://mpi4py.readthedocs.io/en/stable/install.html) |
| `python -m pip install "pymhm[gpu]"` | CuPy and nvmath CUDA solvers | Linux or Windows with a compatible NVIDIA GPU and driver; CUDA 12 packages |
| `python -m pip install "pymhm[meshing]"` | Gmsh, Netgen and meshio | Native library and wheel availability for your platform |
| `python -m pip install "pymhm[visualization]"` | PyVista and VTK plotting | A compatible graphics or offscreen rendering environment |

Combine extras in one installation, for example:

```bash
python -m pip install "pymhm[amg,intel,meshing,visualization]"
```

The GPU extra selects CUDA 12 Python packages; it does not install a GPU driver
or qualify a particular machine. Follow the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
and [nvmath installation guide](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html)
for runtime requirements. The package's [execution guide](execution.md) documents
GPU solver configuration and verified measurements.

DOLFINx, UFL, PETSc/MUMPS and AmgX are separate backend installations. There is
no `fem` or `amgx` pip extra. Installing the `mpi` extra supplies mpi4py. See
[optional capabilities](#optional-capabilities) for PETSc/MUMPS and AmgX
requirements. Notebooks and their datasets are available
in the repository, with an index in the [notebook catalogue](tutorials.md).

### Native UFL assembly

`python -m pip install pymhm` installs the coefficient API and Basix polynomial
core. UFL describes symbolic variational forms; its `fenics-ufl` pip package
does not supply the DOLFINx native mesh, function-space and assembly runtime.
For native UFL applications on Linux, macOS or Windows, create a Conda environment with
compatible DOLFINx packages, then install PyMHM using that environment's Python:

```bash
conda create -n pymhm-fem -c conda-forge python=3.12 "fenics-dolfinx>=0.9,<0.11" pip
conda activate pymhm-fem
python -m pip install pymhm
python -c "import dolfinx, ufl, pymhm; print(dolfinx.__version__, pymhm.__version__)"
```

This follows the [official DOLFINx Conda installation guidance](https://github.com/FEniCS/dolfinx/blob/main/README.md#conda).
On Windows, install Visual Studio's C/C++ compiler and Windows SDK and run from
a developer terminal so the compiler is available for DOLFINx/FFCx JIT.
Upstream describes its native Windows Conda packages as beta; they do not include
PETSc or `petsc4py`. See the [Windows guide](windows.md#native-fem-scope).

For a repository checkout, `pixi run --locked -e fem ...` supplies the native
DOLFINx/UFL stack on Linux, macOS and Windows. Assembly uses native CSR matrices
and vectors, with no PETSc dependency. It requires an MPI runtime and
single-rank local meshes such as `MPI.COMM_SELF`; worker processes own their
native meshes and compiled forms. Choose the numerical solver separately with
`SolverConfig`: SciPy is the default. The locked `fem-intel` profile adds
PARDISO on Linux and Windows; other optional solvers retain their stated
operator and platform requirements. PETSc/MUMPS is included in the Unix FEM
resolution for the optional `petsc` solvers and distributed assembly. Native
execution is qualified separately for each platform in CI. The `introduction`
profile adds the introductory notebook stack; notebook sections using distributed
PETSc references still require PETSc.

The tutorials describe the API in this repository revision. Use the checkout
and its checked-in Pixi lockfile to reproduce the `bind_problem`/`LocalContext`
workflow with the declared native dependencies.

## From a checkout

Install Pixi 0.76.2; both the workspace and the pinned AmgX toolchain require this
version. Check the workspace lockfile before installing an environment:

```bash
pixi --version
pixi list --locked --no-install -e test-core
pixi list --locked --no-install --manifest-path tools/amgx/pixi.toml
pixi install --locked -e test-core
pixi run --locked -e test-core coverage-run
pixi run --locked -e test-core lint
pixi run --locked -e test-core typecheck
pixi run --locked -e docs docs-check
```

The `test-core` environment contains the portable test and development tools.
The lockfile resolves Linux x86-64, Windows x86-64, macOS x86-64 and macOS ARM64
for the portable environments. CI runs on Linux x86-64, Windows x86-64 and macOS
Apple Silicon (ARM64). The macOS x86-64 resolution is available for local checkout
use; it is not an automatically tested CI target. A Linux run does not substitute
for the native Windows/macOS CI jobs.
The core run collects coverage. The independent 99% line and branch gates use
combined Linux core and native FEM measurements; follow the
[coverage procedure](development.md#coverage) to qualify them locally.

```bash
python -m pip install .
# Build an installable wheel and source distribution:
pixi run --locked -e test-core build
```

Use a checkout for development, notebooks and reproducible scientific runs.
The repository includes a Conda recipe and CI that automatically publishes
checked PyPI distributions and documentation on validated release tags.
Building distribution artifacts locally does not publish them. Release
configuration and required repository settings are described in
[development](development.md).

Both release formats provide the complete `pymhm` runtime. The wheel contains
runtime modules, typing files, license and distribution metadata. The source
distribution contains `src/pymhm`, `pyproject.toml`, `README.md`, `LICENSE`, the
backend-required `.gitignore` and generated package metadata, sufficient to build
and install the same runtime.
Documentation, scripts, examples, tests, benchmarks, notebooks, recipes, roadmap
and Pixi files are excluded from both archives. Use the repository checkout for
development, scientific acquisitions and notebooks, including the corresponding
generated figures and field archives when required.

Optional backend adapters are included in the package. Install their dependencies
for the capabilities you use; native runtimes, drivers and solver libraries are
not bundled in the `pymhm` archives. The environments below provide reproducible
combinations, subject to their platform and hardware requirements.

Selected publication figures and the three compact SPE10 input layers are
ordinary Git files. Large computed field archives and intermediate outputs are
generated locally and remain outside Git. Core tests generate small fixtures;
installing the wheel or source distribution does not require previously computed
scientific outputs.

Generate a case with its documented public command. Before using its results,
verify geometry, material and boundary data, approximation spaces, physical norms
and the applicable reference or convergence criteria. A plotting command may
read an existing JSON record; that operation alone does not recompute the physical
field or establish that the record still matches the current implementation.

Execute a notebook directly from a clean checkout, or inspect all declared
acquisitions before computing them:

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run --plan --check
pixi run --locked -e notebooks notebooks-run 72 --study --plan
```

The runner automatically prepares missing current inputs using the complete
public recipes in `scripts/notebook_reproduction.json`. Each command runs in its
locked Pixi environment. Required native notebooks are dispatched to the
declared environment and use that interpreter as their kernel. `--plan` reads
the selected scientific records and lists missing inputs, commands and explicit
source-access limitations without downloading or computing. `--no-prepare`
requires existing current inputs; `--max-data-bytes` selects an explicit archive
budget (three GiB by default).

`--study` executes a complete declared current study even when old inputs are
present. `notebooks-run <selector> --study --plan` lists its exact locked producer
commands, numerical scope and resource requirements first. The machine-readable
`notebooks/catalogue.json` includes these commands for every available study.
The runner expands `{acquisition}` to one fresh UUID shared by all relevant
steps, preserving original attributed result directories.

Original external comparison fields and attributed article images are optional
historical sections. `--historical` requires their original payloads and checksum
identities; fresh PyMHM fields cannot replace them. Standalone analytical controls,
full current-study acquisitions, retained scalar measurements and matched paper
reproductions state their respective scopes. Full Marmousi and high-resolution
reference campaigns require substantial memory, disk and execution time.

Executed copies and a receipt containing source, output and lockfile digests are
written under `build/notebooks`; source notebooks remain unchanged. The complete
historical inventory is available through `scripts/notebook_data.py`. Its raw
`--check` requires all selected original payloads independently of the current
execution mode.

The selected figures in Git support documentation builds from a checkout.
Regenerate and accept figures when changing a scientific case. For example, the
public `gallery-darcy` task computes and plots stated
analytical problems; it does not reproduce a published numerical table. Source
tests and package builds need no computed fields or figures. Scientific notebooks
and the full gallery are executed separately from source-only CI, with their
numerical and rendering checks. CI checks documentation markup; it does not
regenerate the scientific calculations, execute their notebooks or inspect figures.

## Complete native tests on Linux

The `test` environment includes the complete native test stack: DOLFINx,
PETSc/MPI and MUMPS, PARDISO, PyAMG, Gmsh, Netgen, FreeFEM, PyVista/VTK, CuPy,
nvmath and cuDSS. AmgX and its PyAMGX binding use the pinned build toolchain.
The full suite requires two NVIDIA devices and a driver compatible with the
locked CUDA runtime; `test-dependencies` checks these capabilities before tests.

```bash
pixi install --locked -e test
pixi run --locked -e test test-setup-amgx
pixi run --locked -e test test-dependencies
pixi run --locked -e test test-cov
```

The complete and portable suites use all available CPU workers, followed by a
separate phase for tests marked `serial`. Native integration skips in the portable
suite identify capabilities that it does not provide; they do not qualify those
backends. See [development](development.md) for selectors and coverage policy.

In CI, this complete profile runs only when the Tests workflow is manually
dispatched with `full_native` enabled on the configured two-GPU runner. Portable
Core jobs run after the shared lockfile checks; the Integration matrix waits for
every Core job to succeed.
Tests, Lint and Quality, and Docs have dedicated workflows for pull requests and
main-branch pushes; the release workflow reuses their checks before publishing.

## Optional capabilities

| Environment / extra | Purpose | Native requirements |
| --- | --- | --- |
| `pixi run --locked -e fem ...` | UFL/DOLFINx assembly with independently selected solvers | Linux, macOS or Windows; MPI runtime and JIT compiler; optional PETSc/MUMPS on Unix |
| `pixi run --locked -e fem-intel ...` | Native UFL/DOLFINx assembly and PARDISO factors | Linux or Windows x86-64; locked MPI, compiler and Intel MKL stack |
| `pixi run --locked -e introduction ...` | Self-contained introductory notebooks | Locked DOLFINx 0.9, UFL, Basix and notebook stack |
| `pixi run --locked -e meshing ...` | Gmsh, Netgen, meshio | Meshing libraries resolved by Pixi |
| `pixi run --locked -e intel ...` | PARDISO | Intel MKL, supported x86-64 platform |
| `pixi run --locked -e gpu ...` | CuPy QR and cuDSS | NVIDIA device and driver; locked CUDA 12.9 runtime |
| `pixi run --locked -e hpc ...` | MPI, distributed MUMPS and local cuDSS on GPUs | Linux CUDA host; MPI/PETSc and CUDA resolved together |
| `python -m pip install "pymhm[amg]"` | CPU algebraic multigrid | PyAMG |
| `python -m pip install "pymhm[meshing]"` | Import/export and generators | Wheel availability depends on platform |
| `pixi run --locked -e notebooks ...` | Execute notebooks | Jupyter/nbclient and plotting stack |

AmgX uses the optional `pyamgx` bindings and the native NVIDIA AmgX library. The
binding is built against that library; it is not treated as an ordinary
self-contained Python wheel. See [performance](https://github.com/ipes-lncc/pymhm/blob/main/docs/performance.md) for the verified
runtime combinations. PETSc scalar types and available factorization packages
are properties of the installed PETSc build. The `petsc` solver requires MUMPS
for pivoted factorization of local and global saddle matrices. The locked Unix
`fem` environments include it; native DOLFINx assembly with SciPy or PARDISO
needs neither PETSc nor MUMPS. For a custom PETSc installation, enable MUMPS when
building PETSc, for example with `--download-mumps`; installing `petsc4py` alone
does not add a missing native factorization package. Check availability with:

```python
from petsc4py import PETSc

assert PETSc.Sys.hasExternalPackage("mumps")
```

## Development checks

```bash
pixi run --locked -e test-core test
pixi run --locked -e fem test-fem
pixi run --locked -e fem-intel test-fem-portable
pixi run --locked -e meshing test-meshing
pixi run --locked -e notebooks notebooks-run
pixi run --locked -e test-core python -m examples.verify
```

Environment definitions and task names in `pixi.toml` are authoritative. The
metadata checker prevents release-version/dependency drift between manifests.
Run `pixi run --locked -e packaging lock-check` to validate both the workspace
and AmgX toolchain lockfiles without updating their resolutions.
