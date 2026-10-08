# Learn PyMHM: tutorials and notebooks

Start with the [API overview](tutorials/overview.md): define local equations,
couple them through global traces, assemble, solve and reconstruct a field.
Then follow the introductory problems below. Their pages show the mathematical
formulations, executable code, measured outputs and plots directly in the docs.
The ten source notebooks remain available for interactive use. Select a plot
to enlarge it, or use its original image to inspect the field labels and scales.

## Self-contained introductory course

Each tutorial declares its physical data, approximation spaces, local weak
forms and global equations in focused cells. Importable helpers handle field
evaluation, reference comparisons, plotting, archives and performance campaigns.
The primary path expresses the local and global mathematics through executable
UFL weak forms. `MeshHierarchy`, `LocalContext` and `bind_problem` associate
those forms with meshes, spaces and interface representations. Prepared operator
functions are introduced as conveniences. All notebooks are written in English.

Start by editing the parameter cells, then follow the local equations, global
assembly and field comparisons. Helpers are downloaded separately as local `examples` modules and
use the same public numerical API. Spawned workers use importable definitions;
their actual formulation source is shown or linked where it is introduced.
The detailed reference, quadrature and conservation controls remain available
alongside the plots and full campaign settings.

Read Darcy convergence first, then choose a scalar or vector application. The
parallel tutorials assume familiarity with this local/global workflow.

| Problem | Read the rendered tutorial |
| --- | --- |
| Oscillatory Darcy and convergence | [Multiscale Darcy](tutorials/introduction/darcy_multiscale_convergence.md) |
| Reservoir permeability | [Darcy on a SPE10 layer](tutorials/introduction/darcy_spe10_layer.md) |
| Heterogeneous vector elasticity | [Multiscale elasticity](tutorials/introduction/multiscale_elasticity.md) |
| Cell and face moment reconstruction | [MsHHO](tutorials/introduction/mshho_multiscale.md) |
| Independent skeletal spaces | [MH²M](tutorials/introduction/mh2m_multiscale.md) |
| Reaction-dominated local layers | [MHM-USFEM](tutorials/introduction/mhm_usfem_rad.md) |
| Analytical velocity and pressure layers | [Stokes–Brinkman convergence](tutorials/introduction/stokes_brinkman_boundary_layer.md) |
| Parallel local solves and performance | [Darcy speed-up and scalability](tutorials/introduction/darcy_parallel_scalability.md) |
| Spawned processes and complete workflow scaling | [Darcy process scalability](tutorials/introduction/darcy_process_scalability.md) |
| Three-dimensional local AMG and parallel comparisons | [Darcy 3D scalability](tutorials/introduction/darcy_3d_parallel_scalability.md) |

Each page links to its source notebook and gives a command for executing it
with the installed library and a separately downloaded companion, without a clone. Install the notebook and
plotting extras; native UFL examples additionally require the compatible
DOLFINx/UFL backend described in the [installation guide](installation.md).

```bash
python -m pip install 'pymhm[notebooks,visualization]'
jupyter lab darcy_multiscale_convergence.ipynb
```

The examples compare with classical conforming methods and state how their
references are checked: errors against an analytical solution where available,
or differences between refined reference meshes. The parallel tutorials
distinguish the current numerical control from the recorded or optional full
campaign. Analytical solutions and fine numerical references are distinguished.
Macro meshes appear on field panels, and the local and skeletal resolutions are
declared separately.

The [boundary-layer comparisons](cases/introduction-layers.md) distinguish
unresolved RAD profiles from admissible refinement controls and compare the
Brinkman polynomial family with the literature's approximation spaces. Layer
resolution and observed asymptotic rates are reported separately from algebraic
residuals and conservation checks.

For user-defined problems, start with the [variational guide](variational.md)
and the provider notebooks below. They declare local and global equations
through the generic form interface. The introductory physical notebooks put
user-written equations before their comparisons of established formulations.
The main computational paths declare local and global forms. Comparisons and
scientific acquisition helpers also use predefined formulations with their
verified discretizations; archive-only notebooks display the recorded results.

Rendered pages identify their retained validated numerical execution in the
publication manifest. Current standalone instructions are presented
alongside those outputs; running the current notebook produces a separate
receipt for its actual sources and environment. Performance
measurements belong to their recorded hardware and configurations; rendering a
page does not run a new timing campaign. The plots and numerical outputs retain
the notebook's distinction between analytical solutions and classical numerical
references.

The 2D scaling notebooks execute a bounded current numerical control by default
and show the checksum-verified campaign recorded on 2026-10-04 with its original
revision. The historical timing plots retain that provenance; they do not measure
the current revision. Set `PYMHM_RUN_CAMPAIGN=1` to execute the full acquisition
procedure and obtain new strong, weak and crossover measurements.

## Custom spaces and manual definitions

The [custom-interface tutorial](tutorials/custom-interface.md) and its notebook
show how to own the basis, numbering and orientation explicitly. Fully manual
`LocalEquations`/`MultiscaleProblem` records reuse the same numerical owners.
Choose that level for an external convention or a capability not supplied by
the built-in mesh-associated adapters.

## Introductory examples

| Problem | Notebook | Methods |
| --- | --- | --- |
| Darcy: primal locals | [Primal Galerkin](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/primal_galerkin.ipynb) | Declared P2 primal equations; P2/Q2 and 3D formulation comparisons, Dirichlet and Neumann |
| Darcy: mixed locals | [Mixed H(div)](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/mixed_hdiv.ipynb) | Declared RT0/P0 mixed blocks; RT, BDM, enrichment, tensor RT, tetrahedral and prismatic comparisons |
| Darcy: alternative hybrid forms | [Hybrid methods](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/hybrid_methods.ipynb) | Declared three-field MH²M and MsHHO moment blocks; Robin MH and PGMHM comparisons |
| Elasticity | [Primal and mixed methods](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/elasticity/introductory_methods.ipynb) | Displacement Galerkin, displacement–pressure GaLS, weak-symmetry BDM/tensor RT stress |
| Stokes–Brinkman and Oseen | [Flow methods](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/flow/introductory_methods.ipynb) | Taylor–Hood, USFEM and Oseen in 2D/3D |
| Reaction–advection–diffusion | [Transport methods](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/transport/introductory_methods.ipynb) | Galerkin RAD, SUPG and UNUSUAL |
| Complex acoustics | [Helmholtz fields](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/waves/helmholtz/introductory_methods.ipynb) | Complex scalar Helmholtz and interleaved trace coordinates |
| Electromagnetics | [Maxwell trajectory](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/waves/maxwell/introductory_methods.ipynb) | Declared DG mass/curl stages, tangential hybrid traces and staggered time stepping |
| Custom interface representations | [Custom interface](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/custom_interface.ipynb) | Nonorthogonal basis, arbitrary numbering, explicit maps and physical-field equivalence |
| Local/global algebra | [Providers and batches](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/local_global_providers.ipynb) | Primal/mixed providers, physical gauges, external local solver, serial/spawn batches |
| Native assembly and sparse solvers | [DOLFINx/UFL with independent sparse solvers](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/dolfinx_sparse_solvers.ipynb) | Native CSR assembly without PETSc; independent SciPy/PARDISO local/global solves; primal P1 Darcy; integral pressure moments; signed normal-flux traces; serial/spawn execution |
| Native variational assembly | [UFL provider](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/ufl_provider.ipynb) | UFL/DOLFINx local pairings and an independently declared global Equation |
| Vector variational assembly | [Vector UFL](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/vector_ufl.ipynb) | User-written coercive vector reaction-diffusion and oriented traces |
| Recursive variational equations | [Three-level hierarchy](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/variational_hierarchy.ipynb) | Independent A/B/C/D blocks, operator recursion and full-matrix comparison |
| Spatial recursive equations | [Nested Cartesian MHM](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/36_recursive_mhm.ipynb) | Declared Q2/P1 leaf forms, NestedEquations, physical leaf-integral gauges and comparison in identical flat spaces |

The method-family notebooks expose the 28 scalar and 17 vector patch choices.
The provider, vector UFL and hierarchy notebooks introduce user-written forms.
The native primary examples in elasticity, flow, transport and Helmholtz
require the optional native DOLFINx/UFL backend. The runner verifies its
availability before execution. Edit the parameters in a cell and run it to inspect separate
field errors and original-equation diagnostics.
They are analytical patches, not convergence studies or paper reproductions.
The primary transport example uses Galerkin RAD; SUPG and UNUSUAL remain
separate formulation comparisons. The moderate-resistance flow patches do
not qualify extreme Brinkman regimes. The primary Helmholtz example has no
absorption, PML or wave-resolution qualification.
Native UFL execution is reported separately from the Basix-based provider.
The [transport and heat notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/transport/08_transport_and_heat.ipynb)
also declares steady Robin transport and backward Euler equations, with the
same data and spaces used by its predefined-formulation controls.

## Problem folders and detailed studies

The complete [notebook catalogue](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/README.md)
contains all 98 notebooks and the methods used by each. The
[machine-readable index](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/catalogue.json)
uses stable notebook selectors and logical resource paths. The library distribution contains only `pymhm`. Notebook sources and their
verified support ZIPs are separate downloads; small configurations accompany
the support files. Larger data and field
archives remain separate in the [download index](data.md), with source
attribution, checksums and acquisition procedures.

| Folder | Detailed studies |
| --- | --- |
| `introduction/` | Ten self-contained tutorials defining local and global equations, field plots and classical reference comparisons |
| `darcy/` | Primal/mixed convergence, heterogeneous permeability, estimators, MH/MH²M/MsHHO/PGMHM, polygonal and 3D geometry, wells and reservoirs |
| `flow/` | Velocity–pressure examples; `stokes/` and `brinkman_oseen/` contain their focused studies |
| `elasticity/` | Displacement, pressure, stress and rotation; primal/mixed 2D/3D families and reference comparisons |
| `transport/` | Steady RAD, layers, heat and adaptive transient transport |
| `waves/` | `helmholtz/`, `maxwell/` and `elastodynamics/` |
| `foundations/` | Local/global `operators/`, `geometry/` and `general/` examples |
| `convergence/` | The initial cross-problem convergence catalogue |

The [continuous-macroface 3D notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/74_continuous_macrofaces3d.ipynb)
is a self-contained affine Darcy check of continuous, discontinuous and mixed
polynomial face spaces, including independent positive degrees per face.

## Execute downloaded notebooks

Download the `.ipynb` linked from its tutorial or the catalogue, then open it:

```bash
jupyter lab downloaded-notebook.ipynb
```

The first cell explicitly downloads a SHA256-verified companion, adds its local
support directory to the import path and calls the declared preparation helper.
Downloading the archive does not execute code. Its `examples` and `scripts`
sources are inspectable in the printed `ROOT` directory. Numerical algorithms
remain in `pymhm`. The notebook source itself is preserved outside that directory.

For batch execution, extract the companion and run its tool from its workspace:

```bash
python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --timeout 7200
```

The runner uses the active Python and writes an executed copy and receipt to
`build/notebooks`. Set `PYMHM_WORKSPACE` for an explicit writable directory;
the notebook otherwise uses `.pymhm-companions/<SHA256>` below its current directory.
No examples, notebooks, documentation or datasets are installed by pip.

For documentation contributors working in a source checkout, refresh the ten
rendered introductory pages after executing their notebooks:

```bash
pixi run --locked -e introduction tutorials-render
pixi run --locked -e docs docs-check
```

The renderer requires complete executed copies whose cell sources match the
current notebooks. It writes Markdown, plot assets and a digest manifest;
ordinary documentation builds use these saved pages and require no FEM solves
or performance acquisitions.

Inspect the current reproduction plan and its declared inputs before execution:

```bash
python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --plan --check
```

The default plan selects the current executable examples. Use `--study` for
their complete acquisition recipes. Some publication galleries additionally
require historical fields and images; inspect that separate inventory with:

```bash
python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --historical --plan
```

Missing historical inputs are reported explicitly; their absence does not
prevent execution of the current examples. A fine numerical reference
retains its own discretization and refinement uncertainty; it is not an exact
solution. Archived comparisons identify solver provenance and do not execute
external reference programs. See the [case gallery](cases/index.md) for the
separate convergence and scientific acceptance evidence.

## Reusable support and authoring

Notebook cells call the public PyMHM interfaces and import reusable analytical
patch, acquisition and archive routines when needed. Supporting Python modules
remain importable for spawn workers and automated reproducibility; instructional
examples are presented as notebooks rather than command-line scripts. Numerical
formulas have their owners in the package and are not copied between notebooks.

Create a new example in its problem folder, identify its methods and numerical
conventions, and add it to `notebooks/catalogue.json` and `notebooks/README.md`.
Keep large computed fields and executed outputs under `build/` or their declared
scientific archive paths. The [scalar](tutorials/scalar.md),
[vector](tutorials/vector.md) and [provider](tutorials/providers.md) guides explain
spaces and contracts used by these notebooks.
