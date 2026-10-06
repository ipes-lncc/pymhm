# Installation and environments

The core supports Python 3.11–3.13 and requires NumPy, SciPy, threadpoolctl and `fenics-basix>=0.9`.
Basix supplies the built-in polynomial bases without requiring DOLFINx, PETSc or MPI.
Optional native dependencies are loaded only when their functionality is called.

## From a checkout

Install Pixi 0.76.2; both the workspace and the pinned AmgX toolchain require this
version. Check the workspace lockfile before installing an environment:

```bash
pixi --version
pixi list --locked --no-install -e test-core
pixi list --locked --no-install --manifest-path tools/amgx/pixi.toml
pixi install --locked -e test-core
pixi run --locked -e test-core test-cov
pixi run --locked -e test-core lint
pixi run --locked -e test-core typecheck
pixi run -e docs docs-check
```

The `test-core` environment contains the portable test and development tools.
The lockfile resolves Linux x86-64, Windows x86-64, macOS x86-64 and macOS ARM64
for the portable environments. CI runs on Linux x86-64, Windows x86-64 and macOS
Apple Silicon (ARM64). The macOS x86-64 resolution is available for local checkout
use; it is not an automatically tested CI target. A Linux run does not substitute
for the native Windows/macOS CI jobs.

```bash
python -m pip install .
# Build an installable wheel and source distribution:
pixi run --locked -e test-core build
```

Install from a checkout until a release is published. The repository includes
a Conda recipe and CI that automatically publishes checked PyPI distributions
and documentation on validated release tags. Building distribution artifacts
locally does not publish them. Release configuration and required repository
settings are described in [development](development.md).

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

Inspect or validate the field selection for one notebook with:

```bash
pixi run -e notebooks python scripts/notebook_data.py --notebook 23
pixi run -e notebooks python scripts/notebook_data.py --notebook 23 --check
```

The first command lists missing files without downloading anything. The second
requires real local payloads and applies an explicit three-GiB budget. Generate
missing fields with the corresponding public case command. The selectors follow
the versioned JSON records; update `scripts/notebook_data.py` when adding field
reads. The notebook runner checks its selected dependencies before execution.
Existing checksum and numerical checks remain active.

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
| `pixi run -e fem ...` | UFL/DOLFINx assembly, PETSc | Unix FEniCS stack from conda-forge; PETSc with MUMPS |
| `pixi run --locked -e introduction ...` | Self-contained introductory notebooks | Locked DOLFINx 0.9, UFL, Basix and notebook stack |
| `pixi run -e meshing ...` | Gmsh, Netgen, meshio | Meshing libraries resolved by Pixi |
| `pixi run -e intel ...` | PARDISO | Intel MKL, supported x86-64 platform |
| `pixi run -e gpu ...` | CuPy QR and cuDSS | NVIDIA device and driver; locked CUDA 12.9 runtime |
| `pixi run --locked -e hpc ...` | MPI, distributed MUMPS and local cuDSS on GPUs | Linux CUDA host; MPI/PETSc and CUDA resolved together |
| `pip install '.[amg]'` | CPU algebraic multigrid | PyAMG |
| `pip install '.[meshing]'` | Import/export and generators | Wheel availability depends on platform |
| `pixi run -e notebooks ...` | Execute notebooks | Jupyter/nbclient and plotting stack |

AmgX uses the optional `pyamgx` bindings and the native NVIDIA AmgX library. The
binding is built against that library; it is not treated as an ordinary
self-contained Python wheel. See [performance](https://github.com/ipes-lncc/pymhm/blob/main/docs/performance.md) for the verified
runtime combinations. PETSc scalar types and available factorization packages
are properties of the installed PETSc build. The `petsc` solver requires MUMPS
for pivoted factorization of local and global saddle matrices. The locked `fem`
environment includes it. For a custom PETSc installation, enable MUMPS when
building PETSc, for example with `--download-mumps`; installing `petsc4py` alone
does not add a missing native factorization package. Check availability with:

```python
from petsc4py import PETSc

assert PETSc.Sys.hasExternalPackage("mumps")
```

## Development checks

```bash
pixi run --locked -e test-core test
pixi run -e fem test-fem
pixi run --locked -e meshing test-meshing
pixi run -e notebooks notebooks-run
pixi run --locked -e test-core python examples/verify.py
```

Environment definitions and task names in `pixi.toml` are authoritative. The
metadata checker prevents release-version/dependency drift between manifests.
Run `pixi run --locked -e packaging lock-check` to validate both the workspace
and AmgX toolchain lockfiles without updating their resolutions.
