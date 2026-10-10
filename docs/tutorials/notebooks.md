# Notebook catalogue

The [method tutorials](methods/index.md) teach the local/global formulation
step by step. The [Gallery](../gallery/index.md) presents applications and
performance examples. Their pages link to the corresponding English notebooks,
which can be downloaded and run independently of the documentation site.

Use [Getting Started](../getting-started/index.md) to select an environment.
Native UFL examples require a compatible DOLFINx installation; portable
Basix/SciPy examples use the installed core. The [provider guide](../guides/providers.md)
and [custom-interface guide](../guides/custom-interface.md) explain the extension
points for external local solvers and manually declared spaces.

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
contains all 114 notebooks and the methods used by each. The
[machine-readable index](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/catalogue.json)
uses stable notebook selectors and logical resource paths. The library distribution contains only `pymhm`. Notebook sources and their
verified support ZIPs are separate downloads; small configurations accompany
the support files. Larger data and field
archives remain separate in the [download index](../data.md), with source
attribution, checksums and acquisition procedures.

| Folder | Detailed studies |
| --- | --- |
| `introduction/` | Method lessons and application notebooks with local/global equations, fields and reference comparisons |
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

Documentation contributors can render the six application notebooks after
executing their current sources:

```bash
pixi run --locked -e introduction gallery-render
pixi run --locked -e docs docs-check
```

The renderer writes the Gallery notebook pages, plot assets and execution
manifest. The four scalar method lessons are maintained in their canonical
Tutorials pages together with their asymptotic studies; the Gallery does not
publish a second copy of those lessons. Ordinary documentation builds consume
saved pages and do not execute a numerical or performance campaign.

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
external reference programs. See the [case gallery](../cases/index.md) for the
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
scientific archive paths. The [method lessons](methods/index.md) and [provider guide](../guides/providers.md) explain
spaces and contracts used by these notebooks.
