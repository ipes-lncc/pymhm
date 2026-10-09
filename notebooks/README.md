# PyMHM examples by problem

Start with the [API overview](https://ipes-lncc.github.io/pymhm/tutorials/overview/)
and the [rendered method tutorials](https://ipes-lncc.github.io/pymhm/tutorials/).
The foundational applications’ [source notebooks](introduction/README.md) are available for interactive use. Then
use the problem folders and numbered notebooks for additional formulations,
convergence records, geometry variants and literature comparisons. The new
course declares meshes, spaces, material, local forms and global equations in
focused cells. Importable helpers provide reference controls, field evaluation,
plots, archives and performance campaigns. `LocalContext` supplies representation details
while UFL expresses the mathematics; prepared operators follow as conveniences. All tutorials are in English.

For user-written forms, begin with the local/global, UFL, vector UFL and
hierarchy notebooks in `foundations/operators`. They use the generic
`LocalEquations`, `Equation` and `MultiscaleProblem` interface. Introductory
physical notebooks put user-written equations before their method-family
comparisons. The local providers declare mathematical blocks, physical kernels,
moments, trace orientations and boundary loads; method-ready solvers are optional
conveniences. Original-system comparisons assemble the stated equations directly.

The PyMHM distribution contains only `pymhm`. Download a notebook from the
links below or the documentation. Notebooks that use acquisition or plotting companions have a first cell that explicitly downloads a
SHA256-verified companion ZIP containing local `examples` helpers, notebook
execution tools and small configurations. It then prepares only the declared
inputs. These support files are separate from the installed library; you can
inspect or edit them in the printed `ROOT` directory. Numerical algorithms
remain in `pymhm`, and worker definitions remain importable for spawn.

Install the optional notebook/plot dependencies and open the downloaded file:

```bash
python -m pip install 'pymhm[notebooks,visualization]'
jupyter lab downloaded-notebook.ipynb
```

Native DOLFINx/UFL examples additionally require the compatible backend in the
[installation guide](https://ipes-lncc.github.io/pymhm/installation/). NumPy/SciPy
formulations remain independent of it. No clone or Pixi installation is needed.
Downloading the companion does not execute its code. The following explicit
preparation call prints any declared acquisition commands before running them.
Larger inputs remain separate [downloads](https://ipes-lncc.github.io/pymhm/data/),
verified by SHA256 and cached. Generated outputs stay in the writable workspace.

After extracting the companion, run its optional tool from that workspace:

```bash
python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --plan --check
python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --timeout 7200
```

The runner uses the active Python, preserves the downloaded source and writes
executed copies under `build/notebooks`. `--plan` only lists inputs and commands;
`--no-prepare` requires current inputs to exist. `PYMHM_WORKSPACE` selects an
explicit directory; otherwise the first cell uses `.pymhm-companions/<SHA256>`.

`--study` acquires the complete available current scientific campaign with its
stated meshes, spaces, refinement levels and references. Inspect its cost and
commands first with `python -m scripts.run_notebooks /path/to/downloaded-notebook.ipynb --study --plan`.
Fresh output-directory recipes share one automatically expanded acquisition UUID.
The immediate physical example and the complete campaign state their separate
discretizations; an analytical patch is not a substitute for a heterogeneous
or published case.

Original external comparison coefficients and attributed article rasters form
explicitly optional historical sections. `--historical` requires those original
payloads and preserves their checksum checks. Retained JSON measurements
identify their executed revisions; rendering scalar observations does not
recompute a PDE. Current controls, complete current acquisitions and matched
literature reproductions have separate declared scopes. Large full studies can
require substantial memory, storage and computation time.

The machine-readable index is [catalogue.json](catalogue.json).

## Method and strategy workflows

These focused lessons define the physical data, local equations and global equations directly. Standalone Basix/SciPy and UFL workflows import numerical operations from the installed library; they need no application-specific solver helpers. The convergence reader additionally uses attributed records from its verified companion.

| Notebook | Mathematical workflow |
| --- | --- |
| [Mixed-local MHM: flux, pressure and boundary pressure](darcy/mixed_mhm_workflow.ipynb) | explicit RT0/P0 mixed blocks; physical normal-flux binding; fine-cell balance |
| [Robin MH: write the local and global equations](darcy/robin_mh_workflow.ipynb) | UFL Robin volume and boundary forms; Robin multiplier convention; physical pressure and gradient errors |
| [Residual Petrov–Galerkin MHM: jump form and enrichment](darcy/pgmhm_workflow.ipynb) | primal LocalEquations; explicit residual global Equation; constrained residual lift; enriched macro conservation |
| [Independent skeletal refinement for unfitted MHM](darcy/unfitted_trace_workflow.ipynb) | independent local/trace partitions; Basix volume forms; signed common-partition trace integration |
| [Flux recovery, indicators and adaptive refinement](darcy/reconstruction_and_indicators.ipynb) | user-written UFL local/global forms; RT2 moment recovery; four-term energy estimator; Dörfler macro refinement; Oswald potential; RT0 fine-cell equilibration |
| [Qualified method convergence and asymptotic orders](convergence/method_rates.ipynb) | attributed physical error records; space and time refinement; literature hypotheses; successive rate plots |
| [One heterogeneous Darcy problem across CPU, MPI and GPU](darcy/heterogeneous_execution.ipynb) | primal MHM; serial/thread/spawn workers; MPI distributed assembly; CUDA local sparse LU |
| [Generate, exchange and refine material-marked meshes](foundations/geometry/marked_materials.ipynb) | Gmsh physical groups; VTU meshio exchange; exact refinement ancestry |
| [Elastodynamics: equations, Newmark stepping and physical fields](waves/elastodynamics/formulation_workflow.ipynb) | explicit endpoint local/global blocks; Newmark time integration; rigid displacement and velocity controls |
| [GaLS elasticity: displacement-pressure local and global forms](elasticity/gals_mhm_workflow.ipynb) | user-written GaLS stabilized forms; physical pressure convention; elasticity field evaluation |
| [Mixed-stress MHM: stress, displacement and weak rotation](elasticity/mixed_stress_mhm_workflow.ipynb) | mixed H(div) stress forms; weak symmetry; physical traction binding |
| [Oseen MHM: write velocity-pressure and pseudo-traction forms](flow/oseen_variational.ipynb) | UFL Oseen equations; physical half-advection pseudo-traction; velocity and pressure gauges |
| [Transient transport: local forms and time-step global equations](transport/transient_variational.ipynb) | UFL transport forms; backward Euler; OfflineMultiscaleSystem; repeated source stepping |

## introduction

| Notebook | Methods |
| --- | --- |
| [Multiscale Darcy and convergence](introduction/darcy_multiscale_convergence.ipynb) | Explicit primal local operators; quadrilateral MHM; conforming Galerkin; pressure and flux convergence |
| [Parallel Darcy: speed-up and scalability](introduction/darcy_parallel_scalability.ipynb) | Q1 mesh-size sweep from 200×200 to 1000×1000; classical LU and AMG; matched fine-element counts; parallel local solves; complete solve timings; strong/weak scaling |
| [Darcy with spawned processes](introduction/darcy_process_scalability.ipynb) | Explicit Q1 local forms; cross-platform spawn; matched fine-element counts; classical LU/AMG; complete strong/weak scaling |
| [Three-dimensional Darcy: processes and AMG](introduction/darcy_3d_parallel_scalability.ipynb) | Explicit UFL local forms; matched hexahedral fine-cell counts; process strong/weak scaling; CPU AMG and independently measured native GPU condensation |
| [Darcy on a SPE10 layer](introduction/darcy_spe10_layer.ipynb) | Heterogeneous primal locals; quadrilateral MHM; refined conforming Galerkin |
| [Multiscale elasticity](introduction/multiscale_elasticity.ipynb) | Vector UFL local forms; rigid motions; primal MHM; conforming elasticity |
| [MsHHO with an oscillatory coefficient](introduction/mshho_multiscale.ipynb) | Explicit scalar operators; constrained energy reconstruction; cell and face moments; conforming Galerkin |
| [MH²M with an oscillatory coefficient](introduction/mh2m_multiscale.ipynb) | Explicit scalar operators; independent potential and conormal traces; local saddle equations; conforming Galerkin |
| [MHM-USFEM for reaction–diffusion layers](introduction/mhm_usfem_rad.ipynb) | Explicit Galerkin and stabilized local operators; negative residual pairing; analytical layers; classical conforming reference |
| [Stokes–Brinkman boundary-layer convergence](introduction/stokes_brinkman_boundary_layer.ipynb) | Velocity–pressure UFL; Taylor–Hood and USFEM local spaces; analytical layer; refined classical Taylor–Hood |

```bash
jupyter lab darcy_multiscale_convergence.ipynb
```

## convergence

| Notebook | Methods |
| --- | --- |
| [Initial convergence studies](convergence/73_initial_convergence.ipynb) | cross-family convergence catalogue |

## darcy

| Notebook | Methods |
| --- | --- |
| [Start here: Alternative hybrid methods for scalar diffusion](darcy/hybrid_methods.ipynb) | user-written three-field MH2M and MsHHO moment blocks; Robin MH; MH2M; MsHHO; PGMHM |
| [Start here: Darcy with mixed H(div) local problems](darcy/mixed_hdiv.ipynb) | user-written RT0/P0 mixed LocalEquations/Equation; RT0/RT1; BDM/BDM+/BDM++; tensor RT; restricted tetrahedral/prismatic H(div); classical conforming RT1 |
| [Start here: Darcy with primal Galerkin local problems](darcy/primal_galerkin.ipynb) | user-written P2 primal LocalEquations/Equation; primal MHM; local Galerkin P2/Q2; physical Neumann mean gauge |
| [Darcy: primal and mixed convergence](darcy/01_darcy_convergence.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [Darcy across a permeability jump](darcy/02_heterogeneous_darcy.ipynb) | primal MHM; heterogeneous permeability |
| [Independent face partitions and degrees](darcy/03_skeleton_hp.ipynb) | primal MHM; independent hp face traces |
| [Fine-cell conservation in H(div)](darcy/04_mixed_and_reconstructed_flux.ipynb) | primal MHM; mixed RT0/P0 MHM; equilibrated H(div) flux |
| [Published Darcy curves: archived results](darcy/11_published_darcy_2013.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [RT0/P0 Darcy equations and attributed NeoPZ comparisons](darcy/14_neopz_hdiv_comparison.ipynb) | user-written RT0/P0 equations; same-physical-case analytical refinement; retained NeoPZ provenance |
| [BDM2/P1 Darcy: conservative high-order local flux](darcy/17_darcy_bdm.ipynb) | BDM2/P1 mixed Darcy MHM |
| [RT moment reconstruction: normal conformity and continuous-test balance](darcy/18_reconstruction_moments.ipynb) | primal MHM; RT moment reconstruction |
| [Unit-diffusion MHM energy estimator](darcy/20_darcy_estimator.ipynb) | primal MHM; Oswald potential recovery; energy estimator |
| [SPE10 Model 2: physical layers and PyVista](darcy/21_spe10_pyvista.ipynb) | SPE10 material data; PyVista visualization |
| [Quarter-five spot: point wells, layered media and a square obstacle](darcy/22_quarter_five_spot.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [SPE10: published Darcy and Brinkman spaces](darcy/23_spe10_mhm_comparison.ipynb) | quadrilateral primal MHM; USFEM Brinkman |
| [MHM and MsHHO field equivalence](darcy/24_mshho.ipynb) | primal MHM; MsHHO |
| [Enriched rectangular RT elements](darcy/25_tensor_rt.ipynb) | rectangular RT mixed MHM |
| [Three-dimensional tetrahedral MHM](darcy/29_darcy3d.ipynb) | tetrahedral primal MHM |
| [Continuous polynomial macrofaces in three-dimensional MHM](darcy/74_continuous_macrofaces3d.ipynb) | tetrahedral primal MHM; continuous/discontinuous Bernstein face polynomials; mixed continuity and independent degrees |
| [Polygonal macro meshes](darcy/30_polygonal_macro_meshes.ipynb) | polygonal primal MHM |
| [Material-weighted energy estimation](darcy/31_weighted_energy_estimator.ipynb) | primal MHM; material-weighted energy estimator |
| [Raviart–Thomas Darcy: RT0, RT1 and RT2](darcy/33_darcy_rt.ipynb) | RT0; RT1; RT2 |
| [Recursive MHM in identical leaf spaces](darcy/36_recursive_mhm.ipynb) | user-written Q2/P1 NestedEquations and leaf-integral gauges; recursive MHM; direct leaf-space MHM |
| [Periodic Darcy: separate face, local and reference refinement](darcy/37_periodic_darcy.ipynb) | periodic quadrilateral MHM |
| [Material interfaces and skeletal partitions](darcy/38_unfitted_darcy.ipynb) | material-fitted integration; independent skeletal partitions |
| [Adaptive Darcy macro meshes](darcy/39_adaptive_darcy.ipynb) | adaptive primal MHM; weighted energy estimator |
| [Adaptive SPE10 Darcy: published indicators and classical RT2 comparison](darcy/40_spe10_adaptive.ipynb) | adaptive primal MHM; classical RT2/P2 reference |
| [Darcy face-jump indicator](darcy/42_darcy_jump_estimator.ipynb) | primal MHM; pressure jump indicator |
| [Analytical Darcy MHM with full source moments](darcy/43_analytical_darcy.ipynb) | analytical local Darcy MHM |
| [Mapped hexahedral RT1 around a well](darcy/44_mapped_well.ipynb) | mapped hexahedral RT1 MHM |
| [Oscillatory 3D producing-well flow](darcy/47_mapped_oscillatory_well.ipynb) | mapped hexahedral RT1 MHM |
| [Mixed tetrahedral and prismatic Darcy](darcy/51_mixed_well_geometries.ipynb) | tetrahedral and prismatic H(div) Darcy MHM |
| [Three-field multiscale Darcy: MH²M](darcy/52_mh2m.ipynb) | three-field MH2M |
| [Multiscale Hybrid diffusion with Robin local problems](darcy/56_mh.ipynb) | Robin-local Multiscale Hybrid (MH) |
| [Tetrahedral flux reconstruction and adaptive resolution](darcy/58_reconstruction3d.ipynb) | tetrahedral primal MHM; 3D RT moment reconstruction; energy estimator |
| [Petrov–Galerkin MHM on polytopes](darcy/59_pgmhm.ipynb) | Petrov-Galerkin MHM |
| [Material-fitted 3D local meshes and nonuniform face partitions](darcy/63_planar3d.ipynb) | material-fitted tetrahedral MHM; nonuniform face partitions; RT reconstruction |
| [General tetrahedral degree and admissible P5/P2 estimation](darcy/64_tetra_pk.ipynb) | tetrahedral Pk MHM; P5/P2 error estimation |
| [Physical Neumann data and nonconvex macroelements in MH and MH²M](darcy/65_mh_boundary.ipynb) | MH; MH2M; physical Neumann data |
| [Tetrahedral MH and MH²M](darcy/69_mh3d.ipynb) | tetrahedral MH; tetrahedral MH2M |
| [MH²M with independently refined conormals](darcy/70_mh2m_heterogeneous.ipynb) | MH2M; independent pressure and conormal traces |

## elasticity

| Notebook | Methods |
| --- | --- |
| [Start here: Primal and mixed elasticity methods](elasticity/introductory_methods.ipynb) | user-written UFL primal plane elasticity; primal Galerkin; GaLS displacement-pressure; BDM weak-symmetry stress; tensor RT stress |
| [Mixed elasticity and the incompressible limit](elasticity/07_elasticity.ipynb) | user-written P1/P1 GaLS equations and physical rigid/pressure gauges; GaLS elasticity MHM |
| [GaLS elasticity equations and attributed MSL comparisons](elasticity/15_elasticity_reference.ipynb) | user-written P1/P1 GaLS equations; physical pressure identity; retained independent MSL reference |
| [Mixed elasticity: H(div) stress and weak symmetry](elasticity/16_mixed_elasticity.ipynb) | weak-symmetry H(div) elasticity MHM |
| [Mixed H(div) elasticity families](elasticity/28_mixed_elasticity_families.ipynb) | BDM; BDM-plus; BDM-double-plus |
| [General-tensor primal elasticity](elasticity/32_primal_tensor_elasticity.ipynb) | general-tensor primal elasticity MHM |
| [Rectangular RT mixed elasticity](elasticity/34_rectangular_mixed_elasticity.ipynb) | rectangular RT mixed elasticity MHM |
| [General-tensor tetrahedral elasticity](elasticity/35_elasticity3d.ipynb) | tetrahedral general-tensor primal elasticity MHM |
| [HPC4e heterogeneous mixed elasticity](elasticity/50_hpc4e_geomechanics.ipynb) | rectangular RT mixed elasticity MHM |
| [Three-dimensional nearly incompressible elasticity](elasticity/55_gals3d.ipynb) | user-written tetrahedral P2/P2 GaLS equations with physical inverse bound; tetrahedral GaLS elasticity MHM |
| [Three-dimensional weak-symmetry mixed elasticity](elasticity/68_mixed_elasticity3d.ipynb) | 3D Arnold-Falk-Winther mixed elasticity MHM |

## flow

| Notebook | Methods |
| --- | --- |
| [Start here: Stokes–Brinkman and Oseen introductory methods](flow/introductory_methods.ipynb) | user-written UFL P2/P1 Taylor-Hood Brinkman; Taylor-Hood; USFEM; Oseen |
| [Native three-dimensional Stokes, Brinkman and Oseen](flow/54_flow3d.ipynb) | user-written tetrahedral P2/P1 velocity/pressure LocalEquations; 3D Stokes MHM; 3D Brinkman MHM; 3D Oseen MHM |

## flow/brinkman_oseen

| Notebook | Methods |
| --- | --- |
| [Brinkman stabilization and Oseen transport](flow/brinkman_oseen/06_brinkman_and_oseen.ipynb) | user-written USFEM drag0/1/100 and skew Taylor–Hood Oseen blocks; USFEM Brinkman; Oseen MHM |
| [Stabilized Oseen flow and two-level adaptivity](flow/brinkman_oseen/27_oseen_adaptivity.ipynb) | stabilized Oseen MHM; macro and local adaptivity |
| [Stokes–Brinkman: two levels of error and two adaptive strategies](flow/brinkman_oseen/48_stokes_adaptive.ipynb) | Stokes-Brinkman MHM; local and macro adaptivity |

## flow/stokes

| Notebook | Methods |
| --- | --- |
| [The polynomial Stokes problem of Araya et al. (2017)](flow/stokes/05_stokes_published_problem.ipynb) | user-written Taylor–Hood and P1/P1 USFEM LocalEquations; Taylor-Hood MHM; USFEM MHM |
| [Stokes pressure: archived convergence and interface jumps](flow/stokes/12_stokes_pressure_audit.ipynb) | Taylor-Hood MHM; USFEM MHM; conforming DOLFINx reference |
| [Published Stokes experiment: archived high-order results](flow/stokes/13_published_stokes_2017.ipynb) | DOLFINx/UFL local USFEM MHM |

## foundations/general

| Notebook | Methods |
| --- | --- |
| [Visual verification gallery](foundations/general/10_visual_gallery.ipynb) | primal and mixed Darcy; Taylor-Hood and USFEM flow; RAD; heat; elasticity |
| [Higher-order local fields and variable operators](foundations/general/19_native_extensions.ipynb) | user-written Darcy, tensor USFEM, conservative SUPG and backward Euler equations; higher-order primal Darcy; USFEM flow; variable-coefficient RAD; heat |
| [Anisotropic stress and three-dimensional mixed spaces](foundations/general/57_core_extensions.ipynb) | anisotropic mixed elasticity; 3D H(div) Darcy; 3D MsHHO |

## foundations/geometry

| Notebook | Methods |
| --- | --- |
| [Three-dimensional mesh exchange](foundations/geometry/60_mesh_exchange.ipynb) | triangle/polygon/tetrahedron/prism/hexahedron/polyhedron geometry; volume mesh exchange |

## foundations/operators

| Notebook | Methods |
| --- | --- |
| [DOLFINx/UFL with independent sparse solvers](foundations/operators/dolfinx_sparse_solvers.ipynb) | Native CSR assembly without PETSc; independent SciPy/PARDISO local/global solves; primal P1 Darcy; signed normal-flux traces; integral pressure moments; serial/spawn execution |
| [Custom interface spaces](foundations/operators/custom_interface.ipynb) | MeshHierarchy/bind_problem; structural InterfaceSpace/TraceBinding; nonorthogonal bases; manual numbering/orientation; physical-field equivalence |
| [Start here: Local and global forms, providers and ordered batches](foundations/operators/local_global_providers.ipynb) | LocalEquations/Equation; primal and mixed local forms; serial/thread/spawn batches; external local solver |
| [Start here: User-written UFL local and global equations](foundations/operators/ufl_provider.ipynb) | UFL/DOLFINx forms; independent row and column pairings; COMM_SELF local assembly |
| [Start here: Three levels of user-defined equations](foundations/operators/variational_hierarchy.ipynb) | four-block equations; three-level operator recursion; independent full-system comparison |
| [Start here: User-written vector reaction-diffusion equations](foundations/operators/vector_ufl.ipynb) | vector reaction-diffusion; UFL/DOLFINx compilation; oriented trace pairings |
| [Composing local operators and using AMG](foundations/operators/09_local_operators_and_amg.ipynb) | free local operators; hybrid condensation; AMG |

## transport

| Notebook | Methods |
| --- | --- |
| [Start here: Scalar reaction–advection–diffusion methods](transport/introductory_methods.ipynb) | user-written UFL skew Galerkin RAD; RAD Galerkin; SUPG; UNUSUAL |
| [Robin transport and transient heat](transport/08_transport_and_heat.ipynb) | user-written Robin transport and backward Euler LocalEquations; identical-space compatibility controls |
| [Adaptive and Darcy-coupled transient transport](transport/26_adaptive_transient_transport.ipynb) | adaptive RAD; Darcy-coupled backward Euler transport |
| [Three-dimensional conservative RAD: local P4 and face P1](transport/41_rad3d.ipynb) | tetrahedral conservative RAD |
| [Selective RAD kernels and conditioning](transport/45_rad_conditioning.ipynb) | RAD; selective local kernels |
| [Hexagonal boundary-layer problem](transport/46_hexagonal_boundary_layer.ipynb) | hexagonal conservative RAD |
| [Conservative RAD on convex polyhedra](transport/49_polyhedral_rad.ipynb) | polyhedral conservative RAD |
| [Scalar MHM-UNUSUAL reaction–diffusion](transport/53_unusual.ipynb) | scalar reaction-diffusion MHM-UNUSUAL |
| [Nonconvex star-shaped polyhedral RAD](transport/67_star_polyhedra.ipynb) | star-shaped polyhedral conservative RAD |

## waves/elastodynamics

| Notebook | Methods |
| --- | --- |
| [Elastodynamic MHM: spatial and temporal convergence](waves/elastodynamics/71_elastodynamics.ipynb) | Newmark elastodynamic MHM |

## waves/helmholtz

| Notebook | Methods |
| --- | --- |
| [Start here: Complex scalar Helmholtz fields](waves/helmholtz/introductory_methods.ipynb) | user-written UFL real embedding of complex Helmholtz; complex hybrid Helmholtz |
| [Helmholtz MHM: complex fields and PML](waves/helmholtz/61_helmholtz.ipynb) | complex Helmholtz MHM; PML |
| [Marmousi acoustic point source](waves/helmholtz/72_marmousi.ipynb) | Q3 acoustic Helmholtz MHM; classical conforming reference |

## waves/maxwell

| Notebook | Methods |
| --- | --- |
| [Start here: Vector Maxwell fields and a short trajectory](waves/maxwell/introductory_methods.ipynb) | user-written DG mass/curl LocalEquations/Equation stages; DG Maxwell; tangential hybrid traces; staggered time stepping |
| [Time-domain Maxwell MHM in two and three dimensions](waves/maxwell/62_maxwell.ipynb) | 2D and 3D Maxwell MHM |
| [Maxwell photonic device](waves/maxwell/66_maxwell_nanoguide.ipynb) | Cartesian Maxwell MHM; classical numerical reference |
