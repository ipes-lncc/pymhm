# pymhm

Composable Multiscale Hybrid Mixed finite element methods in Python.

`pymhm` couples user-defined local and global variational equations through
explicitly oriented trace coordinates. Declare the forms, choose a local
provider, then call `assemble` or `solve`. Scalar, vector and mixed fields use
the same interface; a local operator can itself be another multiscale problem.
NumPy/SciPy coefficients and optional FEniCS/UFL forms share this contract.
Basix supplies finite-element bases and tabulation.

Version 0.1.0 is research software in pre-alpha development. Built-in workflows
include triangular and polygonal meshes, Cartesian quadrilaterals, tetrahedra,
affine prisms, star-shaped polyhedra and mapped hexahedra. Available equations and approximation
spaces depend on the geometry; the documentation states each verified scope
and distinguishes it from extensions described in the literature.

## Quick start

```bash
pixi install --locked -e test-core
pixi run --locked -e test-core test-cov
```

```python
from pymhm import Equation, LocalEquations, MultiscaleProblem, solve

def local(cell):
    return LocalEquations(
        a=[[2.0]], L=[1.0], b=[[1.0]], c=[[-1.0]],
        d=[[1.0]], dofs=[0],
    )

problem = MultiscaleProblem(
    Equation(0, 0), local, items=[0], trace_size=1, coarse_sizes=(0,),
)
solution = solve(problem)
print(solution.trace, solution.fields)  # lambda = u = 1/3
```

This coefficient example declares $2u+\lambda=1$ and $-u+\lambda=0$.
The [variational guide](docs/variational.md) shows how local and global UFL
forms, retained modes, physical constraints and recursive problems fit this
same interface. It states the supported compilation and elimination limits.
The [vector UFL](notebooks/foundations/operators/vector_ufl.ipynb) and
[three-level hierarchy](notebooks/foundations/operators/variational_hierarchy.ipynb)
notebooks provide direct vector forms and recursive coefficient verification.

The introductory [scalar](docs/tutorials/scalar.md),
[vector](docs/tutorials/vector.md) and [provider](docs/tutorials/providers.md)
tutorials use small executable problems. They cover primal and mixed H(div)
locals, explicit interface conventions, physical gauges and custom local solvers.

Install from a checkout with `python -m pip install .`. Wheel, source distribution,
trusted PyPI publishing and a Conda recipe are provided; no registry publication
is implied by the presence of those files.

The [Windows guide](docs/windows.md) describes native portable-core execution,
SciPy/PyPardiso selection, process workers and installed-wheel checks. The
current native FEM/PETSc profiles use Linux or macOS; WSL2 provides those
environments on a Windows host.

## Verified discretizations and integrations

The predefined physical formulations live in the private `_legacy.models`
package and are imported from their implementation owners. Their numerical
records and stored coefficient contracts document the verified discretizations,
materials and boundary data; they do not qualify arbitrary user-defined forms.

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
- MHM–MsHHO constructions, recursive local problems, material-fitted integration,
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
  actual macro-mesh overlays (`pip install '.[visualization]'` from a checkout).
- SciPy, PETSc, PARDISO, CuPy and cuDSS solvers; CPU PyAMG and optional GPU AmgX.
  AMG local Neumann solves use a compatible SPD complement.
- Deterministic serial/thread/process local execution with bounded batches and
  native thread limits; the coordinator reduces shared-face contributions in
  cell order and retains mesh/DOF metadata for reconstruction.
- Distributed PETSc/MPI assembly, reusable offline operators, resident GPU P1
  batches and augmented CPU/GPU AMG saddle preconditioners.

The [scientific scope](docs/roadmap.md) maps each article to implemented paths,
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

Git contains sources, scientific JSON records and three compact SPE10 input
layers. Large computed fields and figures are generated locally and stay outside
Git. An ordinary package installation and core tests require no archived outputs.
[CONTINUATION.md](CONTINUATION.md) describes how to start from a clean checkout,
run the public calculations and address the remaining scientific work.

Generate only the cases being evaluated, using their documented public commands;
then check their numerical acceptance before plotting or reporting reproduction.
The [notebook catalogue](notebooks/README.md) groups examples by physical problem
and lists their methods. Start with a small analytical example:

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run flow/introductory_methods.ipynb
```

Executed copies retain the problem folders under `build/notebooks`. Historical
numeric IDs also select their notebooks. Analytical patches introduce the API;
they do not replace a published benchmark reproduction.

Before executing a notebook that reads computed fields, inspect its required
local inputs and generate the corresponding calculations:

```bash
pixi run -e notebooks python scripts/notebook_data.py --notebook 68
pixi run -e notebooks python scripts/notebook_data.py --notebook 68 --check
```

Missing fields are reported explicitly. Neither notebooks nor documentation
automatically download computed results. Build the documentation after generating
the figures for its retained cases; CI source tests and package builds do not
depend on those outputs.

## Development

```bash
pixi run --locked -e test-core lint
pixi run --locked -e test-core typecheck
pixi run --locked -e test-core test-cov
pixi run -e fem test-fem
pixi run -e meshing test-meshing
pixi run -e packaging build
pixi run -e packaging check-dist
```

After generating and accepting the selected scientific cases, execute their
notebooks and run `pixi run -e docs docs-check`. Inspect the rendered figures and
MathJax in the browser. Automatic documentation generation remains pending until
its retained scientific cases have a manageable, verified generation workflow.

The portable `test-core` environment supports Linux, Windows and macOS. The
complete `test` environment includes the native CPU and GPU integrations and
requires a Linux CUDA host with two NVIDIA devices. Its pinned AmgX setup and
mandatory dependency checks are described in the [development guide](docs/development.md).
Both suites use all available CPU workers and isolate tests marked `serial`.

CI enforces at least 99% line and branch coverage independently. Executable
notebooks and analytical PDE examples accompany the package. Optional dependency
contracts and actual native-backend integrations are reported separately.

Licensed under LGPL-2.1-only. Citation metadata is in `CITATION.cff`.
