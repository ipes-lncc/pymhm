# Examples and notebooks by problem

All instructional examples are Jupyter notebooks, grouped by the physical
problem. Start with a small analytical example, then select the numbered
studies for convergence, heterogeneous materials and literature comparisons.
Each notebook identifies its methods, local spaces, interface convention and
physical gauge.

For user-defined problems, start with the [variational guide](variational.md)
and the provider notebooks below. They declare local and global equations
through the generic form interface. The introductory physical notebooks put
user-written equations before their comparisons of established formulations.
The main computational paths declare local and global forms. Comparisons and
scientific acquisition helpers also use predefined formulations with their
verified discretizations; archive-only notebooks display the recorded results.

## Introductory examples

| Problem | Notebook | Methods |
| --- | --- | --- |
| Darcy: primal locals | [Primal Galerkin](https://github.com/volpatto/pymhm/blob/main/notebooks/darcy/primal_galerkin.ipynb) | Declared P2 primal equations; P2/Q2 and 3D formulation comparisons, Dirichlet and Neumann |
| Darcy: mixed locals | [Mixed H(div)](https://github.com/volpatto/pymhm/blob/main/notebooks/darcy/mixed_hdiv.ipynb) | Declared RT0/P0 mixed blocks; RT, BDM, enrichment, tensor RT, tetrahedral and prismatic comparisons |
| Darcy: alternative hybrid forms | [Hybrid methods](https://github.com/volpatto/pymhm/blob/main/notebooks/darcy/hybrid_methods.ipynb) | Declared three-field MH²M and MsHHO moment blocks; Robin MH and PGMHM comparisons |
| Elasticity | [Primal and mixed methods](https://github.com/volpatto/pymhm/blob/main/notebooks/elasticity/introductory_methods.ipynb) | Displacement Galerkin, displacement–pressure GaLS, weak-symmetry BDM/tensor RT stress |
| Stokes–Brinkman and Oseen | [Flow methods](https://github.com/volpatto/pymhm/blob/main/notebooks/flow/introductory_methods.ipynb) | Taylor–Hood, USFEM and Oseen in 2D/3D |
| Reaction–advection–diffusion | [Transport methods](https://github.com/volpatto/pymhm/blob/main/notebooks/transport/introductory_methods.ipynb) | Galerkin RAD, SUPG and UNUSUAL |
| Complex acoustics | [Helmholtz fields](https://github.com/volpatto/pymhm/blob/main/notebooks/waves/helmholtz/introductory_methods.ipynb) | Complex scalar Helmholtz and interleaved trace coordinates |
| Electromagnetics | [Maxwell trajectory](https://github.com/volpatto/pymhm/blob/main/notebooks/waves/maxwell/introductory_methods.ipynb) | Declared DG mass/curl stages, tangential hybrid traces and staggered time stepping |
| Local/global algebra | [Providers and batches](https://github.com/volpatto/pymhm/blob/main/notebooks/foundations/operators/local_global_providers.ipynb) | Primal/mixed providers, physical gauges, external local solver, serial/spawn batches |
| Native variational assembly | [UFL provider](https://github.com/volpatto/pymhm/blob/main/notebooks/foundations/operators/ufl_provider.ipynb) | UFL/DOLFINx local pairings and an independently declared global Equation |
| Vector variational assembly | [Vector UFL](https://github.com/volpatto/pymhm/blob/main/notebooks/foundations/operators/vector_ufl.ipynb) | User-written coercive vector reaction-diffusion and oriented traces |
| Recursive variational equations | [Three-level hierarchy](https://github.com/volpatto/pymhm/blob/main/notebooks/foundations/operators/variational_hierarchy.ipynb) | Independent A/B/C/D blocks, operator recursion and full-matrix comparison |
| Spatial recursive equations | [Nested Cartesian MHM](https://github.com/volpatto/pymhm/blob/main/notebooks/darcy/36_recursive_mhm.ipynb) | Declared Q2/P1 leaf forms, NestedEquations, physical leaf-integral gauges and comparison in identical flat spaces |

The method-family notebooks expose the 28 scalar and 17 vector patch choices.
The provider, vector UFL and hierarchy notebooks introduce user-written forms.
The native primary examples in elasticity, flow, transport and Helmholtz
require the Pixi `fem` kernel. They report explicitly when that backend has
not executed. Edit the parameters in a cell and run it to inspect separate
field errors and original-equation diagnostics.
They are analytical patches, not convergence studies or paper reproductions.
The primary transport example uses Galerkin RAD; SUPG and UNUSUAL remain
separate formulation comparisons. The moderate-resistance flow patches do
not qualify extreme Brinkman regimes. The primary Helmholtz example has no
absorption, PML or wave-resolution qualification.
Native UFL execution is reported separately from the Basix-based provider.
The [transport and heat notebook](https://github.com/volpatto/pymhm/blob/main/notebooks/transport/08_transport_and_heat.ipynb)
also declares steady Robin transport and backward Euler equations, with the
same data and spaces used by its predefined-formulation controls.

## Problem folders and detailed studies

The complete [notebook catalogue](https://github.com/volpatto/pymhm/blob/main/notebooks/README.md)
contains all 85 notebooks and the methods used by each. The
[machine-readable index](https://github.com/volpatto/pymhm/blob/main/notebooks/catalogue.json)
uses paths relative to the repository root.

| Folder | Detailed studies |
| --- | --- |
| `darcy/` | Primal/mixed convergence, heterogeneous permeability, estimators, MH/MH²M/MsHHO/PGMHM, polygonal and 3D geometry, wells and reservoirs |
| `flow/` | Velocity–pressure examples; `stokes/` and `brinkman_oseen/` contain their focused studies |
| `elasticity/` | Displacement, pressure, stress and rotation; primal/mixed 2D/3D families and reference comparisons |
| `transport/` | Steady RAD, layers, heat and adaptive transient transport |
| `waves/` | `helmholtz/`, `maxwell/` and `elastodynamics/` |
| `foundations/` | Local/global `operators/`, `geometry/` and `general/` examples |
| `convergence/` | The initial cross-problem convergence catalogue |

## Execute selected notebooks

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run flow/introductory_methods.ipynb
pixi run --locked -e notebooks notebooks-run 01
```

A folder selects its notebooks recursively. Use a qualified path when different
folders contain the same filename. Historical numeric identifiers remain valid.
Executed copies are written to `build/notebooks` with the same subfolders; the
source notebooks remain unexecuted in Git.

Some detailed studies read large locally computed fields and publication images.
Their dependency inventory checks the selected notebooks before execution:

```bash
pixi run --locked -e notebooks python scripts/notebook_data.py --notebook 68
pixi run --locked -e notebooks python scripts/notebook_data.py --notebook darcy --check
```

Missing computed inputs are reported explicitly. A fine numerical reference
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
