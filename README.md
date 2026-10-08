# PyMHM

<p align="center">
  <img src="https://raw.githubusercontent.com/ipes-lncc/pymhm/main/docs/assets/branding/pymhm-logo-readme.png" width="720" alt="PyMHM — Composable Multiscale Hybrid Mixed finite element methods in Python">
</p>

[![Tests](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml/badge.svg)](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/github/ipes-lncc/pymhm/branch/main/graph/badge.svg)](https://app.codecov.io/github/ipes-lncc/pymhm)
[![Lint and Quality](https://github.com/ipes-lncc/pymhm/actions/workflows/lint-and-quality.yml/badge.svg)](https://github.com/ipes-lncc/pymhm/actions/workflows/lint-and-quality.yml)
[![Docs](https://github.com/ipes-lncc/pymhm/actions/workflows/docs.yml/badge.svg)](https://ipes-lncc.github.io/pymhm/)
[![Publish to PyPI](https://github.com/ipes-lncc/pymhm/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/ipes-lncc/pymhm/actions/workflows/publish-pypi.yml)
[![Version: 1.2.0](https://img.shields.io/badge/version-1.2.0-21918c.svg)](https://github.com/ipes-lncc/pymhm/blob/main/pyproject.toml)
[![Python: 3.11–3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab.svg)](https://github.com/ipes-lncc/pymhm/blob/main/pyproject.toml)
[![Supported OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-3776ab.svg)](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml)
[![License: LGPL-2.1-only](https://img.shields.io/badge/license-LGPL--2.1--only-440154.svg)](https://github.com/ipes-lncc/pymhm/blob/main/LICENSE)

`pymhm` couples user-defined local and global variational equations through
explicitly oriented trace coordinates. Declare the forms, choose a local
provider, then call `assemble` or `solve`. Scalar, vector and mixed fields use
the same interface; a local operator can itself be another multiscale problem.
NumPy/SciPy coefficients and optional FEniCS/UFL forms share this contract.
Basix supplies finite-element bases and tabulation.

Version 1.2.0 is an official release of PyMHM. Built-in workflows
include triangular and polygonal meshes, Cartesian quadrilaterals, tetrahedra,
affine prisms, star-shaped polyhedra and mapped hexahedra. Available equations and approximation
spaces depend on the geometry; the documentation states each verified scope
and distinguishes it from extensions described in the literature.

## Installation

Use Python 3.11–3.13 on Linux, macOS or Windows:

```bash
python -m pip install pymhm
```

The base package provides the coefficient interface with NumPy, SciPy and Basix.
Native UFL assembly also requires DOLFINx; installing the symbolic
`fenics-ufl` package with pip does not provide its native assembler. Follow the
[native UFL installation instructions](docs/installation.md#native-ufl-assembly)
for a Conda environment or the locked Pixi `fem` and `introduction` profiles.
Optional extras add CPU AMG,
PARDISO, MPI, CUDA solvers, meshing or visualization; for example,
`python -m pip install "pymhm[amg]"` adds PyAMG.
The [installation guide](docs/installation.md) covers virtual environments,
Windows commands, upgrades and native backend requirements.

## Quick start

Start with the [step-by-step UFL overview](docs/tutorials/overview.md) and the
[introductory course with executed plots](docs/tutorials.md). They connect the
mathematical local and global formulations to the code for scalar, vector and
mixed problems.

Install the notebook dependencies, download a notebook from the catalogue and
open it in Jupyter:

```bash
python -m pip install "pymhm[notebooks]"
jupyter lab primal_galerkin.ipynb
```

The distribution contains only the `pymhm` library. A downloaded notebook's
first cell verifies and extracts its separate companion ZIP of importable helpers
and small configurations into a writable workspace. Its data and recorded fields
use separate verified downloads. This workflow needs no checkout or Pixi. Native UFL examples also need the DOLFINx installation described above.
Use the [locked checkout environments](docs/installation.md#from-a-checkout)
for development and the repository's qualified dependency combinations.

The portable coefficient interface follows the same mesh-to-solution workflow:

```python
import numpy as np
from pymhm import (
    CartesianMacroMesh, Equation, FaceSpace, LocalContext, LocalEquations,
    MeshHierarchy, SkeletonSpace, assemble, bind_interface, bind_problem, solve,
)

macro = CartesianMacroMesh(1, 1)
hierarchy = MeshHierarchy(macro, (macro.submesh(0, 2),))
skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(0) for _ in macro.faces))
interface = bind_interface(skeleton, convention="value")

def local(context: LocalContext) -> LocalEquations:
    """Declare independent volume and interface equations."""
    return context.equations(
        a=[[2.0]], L=[1.0], b=np.ones((1, 4)), c=-np.ones((4, 1)),
        d=np.eye(4),
    )

problem = bind_problem(hierarchy, interface, local, global_equation=Equation(0, 0))
system = assemble(problem)
solution = solve(system)
print(solution.trace, solution.fields)  # every coordinate is 1/6
```

This small algebraic example declares $2u+\sum_F\lambda_F=1$ and
$-u+\lambda_F=0$ on each face. The binding owns the shared numbering and geometric
maps; the user supplies both equations. The
[variational guide](docs/variational.md) describes UFL forms, retained modes,
physical constraints and recursive problems, and states the supported limits.
For complete control, see the [custom-space tutorial](docs/tutorials/custom-interface.md),
which declares a nonorthogonal basis and its independent trial/test maps.

The wheel contains every `pymhm` runtime module, typing and distribution metadata.
The source distribution also carries the README, license and build metadata
needed to rebuild that wheel independently. Neither archive contains notebooks,
example helpers, case registries, configurations, datasets, computed fields,
figures or comparison tools. These repository files have separate
[notebook companions and data downloads](docs/data.md), with declared SHA-256
identities, provenance and terms. Optional native backends require their separate
dependencies.

The [Windows guide](docs/windows.md) describes native portable-core execution,
SciPy/PyPardiso selection, process workers, installed-wheel checks and the
native DOLFINx/UFL profile. Local FEM assembly uses DOLFINx's CSR/vector
interfaces independently of PETSc; choose compatible numerical solvers through
`SolverConfig`. The locked `fem-intel` profile combines FEM assembly with
PARDISO on Linux and Windows. Native Windows requires an MPI runtime and a
Visual Studio JIT compiler. PETSc/MUMPS solvers and distributed PETSc workflows
use the Unix stack, including WSL2 on a Windows host. CI reports qualify each
platform separately.

## Verified discretizations and integrations

The physical formulations compose public finite-element, material, trace and
linear-algebra operations. Importable providers in `examples.formulations`
declare their local and global equations without invoking predefined solvers;
the [variational guide](docs/variational.md) identifies these operations and
their numerical conventions. Existing solver entry points remain available.
Numerical records and stored coefficient contracts document the verified
discretizations, materials and boundary data; they do not qualify arbitrary
user-defined forms.

- Primal Pk, general triangular RT/BDM and enriched rectangular RT Darcy;
  anisotropic permeability, mixed boundary data and physical pure-Neumann gauges.
- Cartesian Qk Darcy, discrete point wells, and SPE10 Model 2 slices with
  explicit units, checked downloads and geometric material-interface integration.
- Taylor–Hood and residual-consistent equal-order USFEM Stokes–Brinkman/Oseen,
  including spatially variable tensor resistance and prescribed convection.
- Displacement–pressure GaLS elasticity through the incompressible limit,
  Taylor–Hood elasticity, and weakly symmetric enriched BDM/RT stress elasticity.
  All local rigid motions, material derivatives and physical constraints are explicit.
- Conservative variable-coefficient RAD with Galerkin or consistent SUPG locals,
  high-order heat diffusion, backward Euler and Darcy/dispersion coupling.
- Polygonal local partitions and star-shaped polyhedra with original polygonal face
  spaces; tetrahedral Pk operators, tetrahedral/prismatic mixed H(div) families,
  and mapped hexahedral RT Darcy fields.
- MsHHO cell/face moment constructions, recursive local problems, material-fitted integration,
  moment reconstruction, and equation-specific error estimation and adaptation.
- MH²M pressure traces, Robin-local MH, residual Petrov–Galerkin MHM and
  unusual reaction–diffusion stabilization, with independent finite element checks.
- Three-dimensional Taylor–Hood/USFEM flow, GaLS elasticity, MsHHO and
  tetrahedral RT reconstruction with dimension-dependent estimator spaces.
- Full anisotropic mixed stress on polygonal macrocells, arbitrary planar
  material cuts and independently controlled normal/interior H(div) orders.
- Three-dimensional weakly symmetric AFW stress elasticity, including the
  incompressible limit, and Newmark MHM elastodynamics with local subcycling.
- Complex Helmholtz waves with absorbing boundaries and perfectly matched layers;
  transient Maxwell TM and three-dimensional vector fields with explicit CFL controls.
- Independent face partitions and degrees, including continuous subface traces;
  physical local kernel constraints, source/trace factorization reuse and
  conservative RT0 flux equilibration.
- User-defined UFL local operators, explicit local/global form descriptions and
  callable providers; optional mesh import/export with physical tags.
- Basix reference elements, nodal simplex/tensor bases and RT/BDM tabulation,
  with explicit native basis ordering, transformations and coefficient digests.
- PyVista/VTK export and rendering, including broken finite-element fields and
  actual macro-mesh overlays (`python -m pip install "pymhm[visualization]"`).
- SciPy, PETSc, PARDISO, CuPy and cuDSS solvers; CPU PyAMG and optional GPU AmgX.
  AMG local Neumann solves use a compatible SPD complement.
- Deterministic serial/thread/process local execution with bounded batches and
  native thread limits; the coordinator reduces shared-face contributions in
  cell order and retains mesh/DOF metadata for reconstruction.
- Distributed PETSc/MPI assembly, reusable offline operators, resident GPU P1
  batches and augmented CPU/GPU AMG saddle preconditioners.

The [scientific scope](ROADMAP.md#scientific-scope-and-acceptance-criteria) maps each article to implemented paths,
verification evidence and remaining reproduction limits. The stated geometry,
space compatibility and estimator hypotheses are part of each capability.
See [theory](docs/theory.md),
[literature](docs/literature.md), [verification](docs/verification.md) and
[performance](docs/performance.md) for assumptions and evidence.

The [visual case gallery](docs/cases/index.md) compares computed fields with
analytical references, error maps, profiles and convergence expectations.
The [MSL field comparison](docs/cases/reference-comparison.md) checks primal
Darcy fields against MSL_MHM with MSL_CG and MSL_Core. The
[NeoPZ comparison](docs/cases/neopz.md) checks RT0/P0 fields against
[NeoPZ](https://github.com/labmec/neopz) and distinguishes that assembly from
Labmec/MHM's higher-order controller. Each case records the actual spaces and
boundary conventions used in the comparison.
The [GaLS elasticity comparison](docs/cases/elasticity-reference.md) matches
MSL fields over five mesh levels and three local polynomial degrees.
The [SPE10 case](docs/cases/spe10.md) identifies the published layers, spaces and
boundary conditions and compares pressure profiles directly with article data.
The [quarter-five-spot case](docs/cases/quarter-five-spot.md) includes point wells,
an analytical series, and MSL/NeoPZ comparisons for a low-permeability obstacle.

## Reproducible calculations

Git contains sources, scientific JSON records, three compact SPE10 input layers
and the selected figures used by the published documentation. Large computed
fields and intermediate outputs are generated locally and stay outside Git.
The installed distribution contains only the `pymhm` library. Notebook sources,
companions and acquisition metadata are separate repository and website files.
Selected data and recorded figures use verified downloads; current fields and
intermediate outputs are generated in the user's working directory. An
ordinary package installation and core tests require no computed field archives.
[ROADMAP.md](ROADMAP.md) sets implementation and validation priorities,
dependencies and acceptance criteria for the next milestones.

Generate only the cases being evaluated, using their documented public commands;
then check their numerical acceptance before plotting or reporting reproduction.
The [notebook catalogue](notebooks/README.md) groups examples by physical problem
and lists their methods. Start with a small analytical example:

```bash
python -m scripts.run_notebooks darcy/primal_galerkin.ipynb
python -m scripts.run_notebooks flow/introductory_methods.ipynb
# The same examples in the locked checkout environment:
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run flow/introductory_methods.ipynb
```

Executed copies retain the problem folders under `build/notebooks`. Historical
numeric IDs also select their notebooks. Analytical patches introduce the API;
they do not replace a published benchmark reproduction.

Inspect the declared calculations before a complete study:

```bash
python -m scripts.run_notebooks 68 --study --plan
```

Current inputs are prepared by their public recipes using the active Python
environment. Pinned raw datasets are downloaded only when the selected case
needs them; checksum and size conventions remain explicit. Complete studies
retain their original cost and numerical scope. Historical external field archives
require their original supplied payloads and are identified separately. Regenerating
a scientific case or replaying its fields requires its documented calculations;
package builds and core tests do not require computed field archives.

## Development

```bash
pixi run --locked -e test-core lint
pixi run --locked -e test-core typecheck
pixi run --locked -e test-core coverage-run
pixi run --locked -e fem test-fem
pixi run --locked -e fem-intel test-fem-portable
pixi run --locked -e meshing test-meshing
pixi run --locked -e packaging build
pixi run --locked -e packaging check-dist
```

After generating and accepting the selected scientific cases, execute their
notebooks and run `pixi run --locked -e docs docs-check`. Inspect the rendered figures and
MathJax in the browser. CI checks documentation markup; scientific acquisition,
notebook execution and full gallery generation require their separate acceptance.

The portable `test-core` environment supports Linux, Windows and macOS. The
complete `test` environment includes the native CPU and GPU integrations and
requires a Linux CUDA host with two NVIDIA devices. Its pinned AmgX setup and
mandatory dependency checks are described in the [development guide](docs/development.md).
Both suites use all available CPU workers and isolate tests marked `serial`.

CI tests the portable core on Linux x86-64, Windows x86-64 and macOS Apple
Silicon (ARM64). It enforces independent 99% line and branch coverage gates on
the combined Linux core and native FEM measurements from the same revision.
The [development guide](docs/development.md#coverage) provides the local commands.
Executable notebooks and their companion helpers are separate repository and
website downloads. Optional
dependency contracts and actual native-backend integrations are reported separately.

Dedicated [Tests](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml),
[Lint and Quality](https://github.com/ipes-lncc/pymhm/actions/workflows/lint-and-quality.yml)
and [Docs](https://github.com/ipes-lncc/pymhm/actions/workflows/docs.yml) workflows
run independently on pull requests and main-branch pushes. The complete two-GPU
suite is opt-in through manual dispatch. On a validated release tag,
[Publish to PyPI](https://github.com/ipes-lncc/pymhm/actions/workflows/publish-pypi.yml)
validates metadata and runs Tests, Quality and Docs checks in parallel, then
publishes the checked distributions to PyPI, creates the GitHub Release and
deploys the documentation to GitHub Pages.
The Docs workflow also supports manual publication from `main` or a release tag
by enabling its `publish` input.
Maintainers configure the PyPI trusted publisher for `publish-pypi.yml`, Pages
with GitHub Actions as its source and the `github-pages` environment to accept
release tags; see [development](docs/development.md).

Prepare release notes and synchronize current versions with
`pixi run --locked -e release release-prepare VERSION`, after running
`release-fetch` in the same environment. The task uses git-cliff and preserves
handwritten notes. Official releases use `X.Y.Z` versions and matching `vX.Y.Z`
tags; for example, prepare `1.1.0` and tag `v1.1.0`. Use `--initial` only when no
canonical release tag exists in the fetched main-branch history.

Licensed under LGPL-2.1-only. Citation metadata is in `CITATION.cff`.

## Institutional Support

PyMHM is developed by the
[Innovative Parallel numErical Solvers (IPES)](https://ipes.lncc.br/)
research group and receives institutional support from the
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br),
a research unit of the
[Ministério da Ciência, Tecnologia e Inovação (MCTI)](https://www.gov.br/mcti/pt-br),
Brazil.

<p align="center">
  <a href="https://ipes.lncc.br/">
    <img src="https://raw.githubusercontent.com/ipes-lncc/pymhm/main/docs/assets/institutions/ipes.png" width="300" alt="IPES — Innovative Parallel numErical Solvers">
  </a>
  <a href="https://www.gov.br/lncc/pt-br">
    <img src="https://raw.githubusercontent.com/ipes-lncc/pymhm/main/docs/assets/institutions/lncc-readme.svg" width="190" alt="LNCC — Laboratório Nacional de Computação Científica">
  </a>
  <a href="https://www.gov.br/mcti/pt-br">
    <img src="https://raw.githubusercontent.com/ipes-lncc/pymhm/main/docs/assets/institutions/mcti-readme.svg" width="270" alt="MCTI — Ministério da Ciência, Tecnologia e Inovação">
  </a>
</p>
